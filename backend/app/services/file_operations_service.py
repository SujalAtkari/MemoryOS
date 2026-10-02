import hashlib
import json
import os
import re
import stat
import sys
from datetime import datetime
from pathlib import Path, PureWindowsPath

from app.services.hashing_service import is_supported_image
from app.services.index_storage import save_json_atomically, serialize_index_update
from app.services.library_service import INDEX_PATH

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tiff', '.tif'}
_INVALID_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED_WINDOWS_NAME = re.compile(
    r'^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?$',
    re.IGNORECASE,
)
_DUPLICATE_FIELDS = (
    'duplicate_group_id',
    'duplicate_type',
    'duplicate_confidence',
    'duplicate_match_method',
    'duplicate_match_distance',
    'duplicate_updated_at',
    'duplicate_groups',
)


class FileOperationError(ValueError):
    pass


def _load_index() -> dict:
    if not INDEX_PATH.exists():
        raise FileOperationError('The local library index does not exist.')
    try:
        with INDEX_PATH.open('r', encoding='utf-8') as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FileOperationError('The local library index could not be read.') from exc
    if not isinstance(payload, dict) or not isinstance(payload.get('libraries'), dict):
        raise FileOperationError('The local library index has an invalid structure.')
    return payload


def _save_index(payload: dict) -> None:
    try:
        save_json_atomically(INDEX_PATH, payload)
    except OSError as exc:
        raise RuntimeError(
            'The file operation succeeded, but the index could not be saved. '
            'Rescan the library to reconcile the index.'
        ) from exc


def _indexed_library(index: dict, library_id: str) -> tuple[dict, Path]:
    library = index['libraries'].get(library_id)
    if not isinstance(library, dict):
        raise FileOperationError('Indexed library was not found.')
    library_root_value = library.get('library_root')
    if not isinstance(library_root_value, str) or not library_root_value.strip():
        raise FileOperationError('Indexed library has no valid root path.')
    try:
        library_root = Path(library_root_value).resolve(strict=True)
    except OSError as exc:
        raise FileOperationError('Indexed library root is unavailable.') from exc
    if not library_root.is_dir():
        raise FileOperationError('Indexed library root is not a directory.')
    if not isinstance(library.get('records'), dict):
        raise FileOperationError('Indexed library has an invalid record collection.')
    return library, library_root


def _relative_parts(relative_path: object) -> tuple[str, ...]:
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise FileOperationError('Indexed image path is invalid.')
    windows_path = PureWindowsPath(relative_path)
    if windows_path.is_absolute() or windows_path.drive or relative_path.startswith(('\\\\', '//')):
        raise FileOperationError('Absolute or network paths are not allowed.')
    normalized = relative_path.replace('\\', '/')
    parts = tuple(normalized.split('/'))
    if any(part in ('', '.', '..') for part in parts):
        raise FileOperationError('Indexed image path contains an unsafe path component.')
    return parts


def _is_link_or_junction(path: Path) -> bool:
    is_junction = getattr(path, 'is_junction', None)
    return path.is_symlink() or bool(is_junction and is_junction())


def _contained_path(root: Path, relative_path: object, *, must_exist: bool = True) -> Path:
    parts = _relative_parts(relative_path)
    candidate = root.joinpath(*parts)
    try:
        resolved = candidate.resolve(strict=must_exist)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise FileOperationError('Image path is missing or outside its indexed library.') from exc
    current = root
    for part in parts:
        current = current / part
        if _is_link_or_junction(current):
            raise FileOperationError('Symbolic links and reparse-point paths are not allowed.')
    return resolved


def _find_record(index: dict, library_id: str, record_id: str) -> tuple[dict, Path, str, dict]:
    if not isinstance(record_id, str) or not record_id:
        raise FileOperationError('Record ID is required.')
    library, root = _indexed_library(index, library_id)
    records = library['records']
    relative_path = record_id if record_id in records else None
    if relative_path is None:
        matching = [
            key for key, value in records.items()
            if isinstance(value, dict) and value.get('record_id') == record_id
        ]
        if len(matching) != 1:
            raise FileOperationError('Indexed image record was not found or is ambiguous.')
        relative_path = matching[0]
    record = records[relative_path]
    if not isinstance(record, dict):
        raise FileOperationError('Indexed image record is invalid.')
    if record.get('file_status') == 'missing':
        raise FileOperationError('Indexed image is marked missing. Rescan the library before operating on it.')
    stored_path = record.get('relative_path', relative_path)
    if stored_path != relative_path:
        raise FileOperationError('Indexed image path does not match its record key.')
    source = _contained_path(root, relative_path)
    if not source.is_file() or not is_supported_image(str(source)):
        raise FileOperationError('Indexed source is not an existing supported image file.')
    return library, root, relative_path, record


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_indexed_file(source: Path, record: dict) -> None:
    indexed_hash = record.get('sha256')
    if isinstance(indexed_hash, str) and re.fullmatch(r'[0-9a-fA-F]{64}', indexed_hash):
        if _sha256(source).lower() != indexed_hash.lower():
            raise FileOperationError(
                'Image has changed since indexing. Rescan the library before modifying it.'
            )
        return
    metadata_available = False
    file_size = record.get('file_size')
    if isinstance(file_size, int) and not isinstance(file_size, bool):
        metadata_available = True
        if source.stat().st_size != file_size:
            raise FileOperationError(
                'Image has changed since indexing. Rescan the library before modifying it.'
            )
    modified_time = record.get('modified_time')
    if isinstance(modified_time, str):
        try:
            indexed_timestamp = datetime.fromisoformat(modified_time).timestamp()
        except ValueError:
            indexed_timestamp = None
        if indexed_timestamp is not None:
            metadata_available = True
            if int(source.stat().st_mtime) != int(indexed_timestamp):
                raise FileOperationError(
                    'Image has changed since indexing. Rescan the library before modifying it.'
                )
    if not metadata_available:
        raise FileOperationError(
            'Indexed image has no usable integrity metadata. Rescan the library before modifying it.'
        )


def _ensure_supported_source(
    root: Path,
    relative_path: str,
    record: dict,
    *,
    verify_integrity: bool = True,
) -> Path:
    source = _contained_path(root, relative_path)
    if not source.is_file() or not is_supported_image(str(source)):
        raise FileOperationError('Indexed source is not an existing supported image file.')
    if verify_integrity:
        _validate_indexed_file(source, record)
    return source


def _save_after_filesystem_change(index: dict, library: dict, records: dict) -> None:
    _invalidate_duplicate_results(library, records)
    _save_index(index)


def _invalidate_duplicate_results(library: dict, records: dict) -> None:
    library.pop('duplicate_detection', None)
    for record in records.values():
        if isinstance(record, dict):
            for field in _DUPLICATE_FIELDS:
                record.pop(field, None)


def _response(record_id: str, filename: str, relative_path: str, message: str) -> dict:
    return {
        'success': True,
        'record_id': record_id,
        'filename': filename,
        'relative_path': relative_path,
        'message': message,
    }


def _mutation_response(
    old_record_id: str,
    filename: str,
    relative_path: str,
    record: dict,
    message: str,
) -> dict:
    result = _response(relative_path, filename, relative_path, message)
    result['previous_record_id'] = old_record_id
    public_record_fields = (
        'record_id',
        'filename',
        'relative_path',
        'extension',
        'file_size',
        'modified_time',
        'width',
        'height',
        'aspect_ratio',
        'sha256',
        'phash',
        'processing_status',
        'file_status',
    )
    result['record'] = {
        field: record[field]
        for field in public_record_fields
        if field in record
    }
    return result


def _open_native(path: Path) -> None:
    if sys.platform != 'win32':
        raise FileOperationError('Opening files through the OS is currently supported on Windows only.')
    try:
        os.startfile(str(path))
    except OSError as exc:
        raise FileOperationError('The operating system could not open the requested item.') from exc


def open_indexed_image(library_id: str, record_id: str) -> dict:
    index = _load_index()
    _library, root, relative_path, record = _find_record(index, library_id, record_id)
    source = _ensure_supported_source(
        root, relative_path, record, verify_integrity=False
    )
    _open_native(source)
    return _response(record_id, source.name, relative_path, 'Image opened.')


def get_indexed_image_preview(library_id: str, record_id: str) -> Path:
    index = _load_index()
    _library, root, relative_path, record = _find_record(index, library_id, record_id)
    return _ensure_supported_source(
        root, relative_path, record, verify_integrity=False
    )


def open_image_folder(library_id: str, record_id: str) -> dict:
    index = _load_index()
    _library, root, relative_path, record = _find_record(index, library_id, record_id)
    source = _ensure_supported_source(
        root, relative_path, record, verify_integrity=False
    )
    _open_native(source.parent)
    return _response(record_id, source.name, relative_path, 'Containing folder opened.')


def list_library_folders(library_id: str, record_id: str) -> dict:
    index = _load_index()
    _library, root, _relative_path, _record = _find_record(index, library_id, record_id)
    folders = ['']
    for current_root, dir_names, _file_names in os.walk(root, followlinks=False):
        current_path = Path(current_root)
        safe_dirs = []
        for name in sorted(dir_names):
            candidate = current_path / name
            if _is_link_or_junction(candidate):
                continue
            try:
                candidate.resolve(strict=True).relative_to(root)
            except (OSError, RuntimeError, ValueError):
                continue
            safe_dirs.append(name)
            folders.append(candidate.relative_to(root).as_posix())
        dir_names[:] = safe_dirs
    return {'library_id': library_id, 'folders': folders}


def _validate_filename(filename: str, current_extension: str) -> str:
    if not isinstance(filename, str) or not filename.strip():
        raise FileOperationError('A new filename is required.')
    if filename != filename.strip() or filename in ('.', '..') or _INVALID_FILENAME.search(filename):
        raise FileOperationError('Filename contains invalid characters or path components.')
    if filename.endswith(('.', ' ')) or _RESERVED_WINDOWS_NAME.fullmatch(filename):
        raise FileOperationError('Filename is reserved or invalid on Windows.')
    supplied_extension = Path(filename).suffix.lower()
    if supplied_extension:
        if supplied_extension not in IMAGE_EXTENSIONS:
            raise FileOperationError('New filename must keep or use a supported image extension.')
        return filename
    return filename + current_extension


def _move_no_overwrite(source: Path, destination: Path) -> None:
    if os.path.lexists(destination):
        raise FileOperationError('Destination already exists; no file was overwritten.')
    try:
        if sys.platform == 'win32':
            os.rename(source, destination)
        else:
            os.link(source, destination)
            try:
                os.unlink(source)
            except OSError:
                destination.unlink(missing_ok=True)
                raise
    except FileOperationError:
        raise
    except OSError as exc:
        if os.path.lexists(destination):
            raise FileOperationError('Destination already exists; no file was overwritten.') from exc
        raise FileOperationError('Filesystem operation failed; the index was not changed.') from exc


def _update_record_path(records: dict, old_path: str, new_path: str, record: dict, destination: Path) -> None:
    record['record_id'] = new_path
    record['relative_path'] = new_path
    record['filename'] = destination.name
    record['extension'] = destination.suffix.lower()
    file_stats = destination.stat()
    record['file_size'] = file_stats.st_size
    record['modified_time'] = datetime.fromtimestamp(file_stats.st_mtime).isoformat(timespec='seconds')
    record['file_status'] = 'available'
    records.pop(old_path)
    records[new_path] = record


@serialize_index_update
def rename_indexed_image(library_id: str, record_id: str, new_filename: str) -> dict:
    index = _load_index()
    library, root, relative_path, record = _find_record(index, library_id, record_id)
    source = _ensure_supported_source(root, relative_path, record)
    filename = _validate_filename(new_filename, source.suffix)
    destination = source.with_name(filename)
    if destination == source:
        raise FileOperationError('New filename is unchanged.')
    destination_relative = destination.relative_to(root).as_posix()
    if destination.exists():
        raise FileOperationError('Destination already exists; no file was overwritten.')
    _move_no_overwrite(source, destination)
    try:
        _update_record_path(library['records'], relative_path, destination_relative, record, destination)
        _save_after_filesystem_change(index, library, library['records'])
    except RuntimeError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise RuntimeError(
            'Image was renamed, but the index update failed. Rescan the library to reconcile it.'
        ) from exc
    return _mutation_response(
        record_id,
        destination.name,
        destination_relative,
        record,
        'Image renamed.',
    )


def _destination_folder(root: Path, relative_folder: str) -> Path:
    if relative_folder == '':
        return root
    parts = _relative_parts(relative_folder)
    destination = _contained_path(root, '/'.join(parts))
    if not destination.is_dir():
        raise FileOperationError('Destination folder does not exist inside the indexed library.')
    return destination


@serialize_index_update
def move_indexed_image(
    library_id: str,
    record_id: str,
    destination_folder: str,
) -> dict:
    if not isinstance(destination_folder, str):
        raise FileOperationError('A library-relative destination folder is required.')
    index = _load_index()
    library, root, relative_path, record = _find_record(index, library_id, record_id)
    source = _ensure_supported_source(root, relative_path, record)
    folder = _destination_folder(root, destination_folder)
    destination = folder / source.name
    if destination == source:
        raise FileOperationError('Image is already in the selected folder.')
    if destination.exists():
        raise FileOperationError('Destination already exists; no file was overwritten.')
    destination_relative = destination.relative_to(root).as_posix()
    _move_no_overwrite(source, destination)
    try:
        _update_record_path(library['records'], relative_path, destination_relative, record, destination)
        _save_after_filesystem_change(index, library, library['records'])
    except RuntimeError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise RuntimeError(
            'Image was moved, but the index update failed. Rescan the library to reconcile it.'
        ) from exc
    return _mutation_response(
        record_id,
        destination.name,
        destination_relative,
        record,
        'Image moved.',
    )


@serialize_index_update
def delete_indexed_image(library_id: str, record_id: str, confirm: bool) -> dict:
    if confirm is not True:
        raise FileOperationError('Explicit confirmation is required to delete the original image.')
    index = _load_index()
    library, root, relative_path, record = _find_record(index, library_id, record_id)
    source = _ensure_supported_source(root, relative_path, record)
    file_stats = source.lstat()
    if not stat.S_ISREG(file_stats.st_mode):
        raise FileOperationError('Only regular image files can be deleted.')
    try:
        source.unlink()
    except OSError as exc:
        raise FileOperationError('Image could not be deleted; the index was not changed.') from exc
    record['file_status'] = 'missing'
    record['processing_status'] = 'pending'
    try:
        _save_after_filesystem_change(index, library, library['records'])
    except RuntimeError:
        raise
    return _mutation_response(
        record_id,
        source.name,
        relative_path,
        record,
        'Original image deleted and marked missing in the index.',
    )
