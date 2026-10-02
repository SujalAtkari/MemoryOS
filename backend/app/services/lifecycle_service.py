import hashlib
import json
import os
import re
import stat
from datetime import datetime
from pathlib import Path, PureWindowsPath

from app.services.hashing_service import is_supported_image
from app.services.file_operations_service import _invalidate_duplicate_results
from app.services.index_storage import save_json_atomically, serialize_index_update
from app.services.library_service import INDEX_PATH

class LifecycleError(ValueError):
    pass


def _load_index() -> dict:
    if not INDEX_PATH.exists():
        return {'version': 1, 'libraries': {}}
    try:
        with INDEX_PATH.open('r', encoding='utf-8') as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LifecycleError('The local library index could not be read.') from exc
    if not isinstance(payload, dict) or not isinstance(payload.get('libraries'), dict):
        raise LifecycleError('The local library index has an invalid structure.')
    return payload


def _save_index(payload: dict) -> None:
    try:
        save_json_atomically(INDEX_PATH, payload)
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeError('Lifecycle changes could not be saved to the index.') from exc


def _safe_relative_parts(value: object) -> tuple[str, ...]:
    if not isinstance(value, str) or not value.strip():
        raise LifecycleError('Relative image path is invalid.')
    windows_path = PureWindowsPath(value)
    if windows_path.is_absolute() or windows_path.drive or value.startswith(('\\\\', '//')):
        raise LifecycleError('Absolute and network paths are not allowed.')
    parts = tuple(value.replace('\\', '/').split('/'))
    if any(part in ('', '.', '..') for part in parts):
        raise LifecycleError('Relative image path contains an unsafe path component.')
    return parts


def _is_link_or_junction(path: Path) -> bool:
    is_junction = getattr(path, 'is_junction', None)
    return path.is_symlink() or bool(is_junction and is_junction())


def _resolve_inside(root: Path, relative_path: object, *, strict: bool) -> Path:
    parts = _safe_relative_parts(relative_path)
    candidate = root.joinpath(*parts)
    try:
        resolved = candidate.resolve(strict=strict)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise LifecycleError('Path is missing or outside the indexed library.') from exc
    current = root
    for part in parts:
        current = current / part
        if _is_link_or_junction(current):
            raise LifecycleError('Symbolic links and reparse-point paths are not allowed.')
    return resolved


def _get_library(index: dict, library_id: str) -> tuple[dict, Path]:
    library = index['libraries'].get(library_id)
    if not isinstance(library, dict):
        raise LifecycleError('Indexed library was not found.')
    root_value = library.get('library_root')
    if not isinstance(root_value, str) or not root_value.strip():
        raise LifecycleError('Indexed library root is invalid.')
    try:
        root = Path(root_value).resolve(strict=True)
    except OSError as exc:
        raise LifecycleError('Indexed library root is unavailable.') from exc
    if not root.is_dir() or not isinstance(library.get('records'), dict):
        raise LifecycleError('Indexed library or its records are invalid.')
    return library, root


def _select_records(records: dict, record_id: str | None) -> list[tuple[str, dict]]:
    valid = [(key, value) for key, value in records.items() if isinstance(key, str) and isinstance(value, dict)]
    if record_id is None:
        return valid
    if record_id in records and isinstance(records[record_id], dict):
        return [(record_id, records[record_id])]
    matches = [(key, value) for key, value in valid if value.get('record_id') == record_id]
    if len(matches) != 1:
        raise LifecycleError('Indexed image record was not found or is ambiguous.')
    return matches


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: object) -> str | None:
    if isinstance(value, str) and re.fullmatch(r'[0-9a-fA-F]{64}', value):
        return value.lower()
    return None


def _time_value(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).isoformat(timespec='seconds')


def _stat_matches(record: dict, size: int, modified_time: str) -> bool:
    expected_size = record.get('lifecycle_observed_size', record.get('file_size'))
    expected_time = record.get('lifecycle_observed_modified_time', record.get('modified_time'))
    if not isinstance(expected_size, int) or isinstance(expected_size, bool):
        return False
    if not isinstance(expected_time, str):
        return False
    try:
        expected_timestamp = datetime.fromisoformat(expected_time).timestamp()
        actual_timestamp = datetime.fromisoformat(modified_time).timestamp()
    except ValueError:
        return False
    return expected_size == size and int(expected_timestamp) == int(actual_timestamp)


def _record_info(relative_path: str, record: dict, status: str) -> dict:
    return {
        'record_id': record.get('record_id', relative_path),
        'filename': record.get('filename', Path(relative_path).name),
        'relative_path': relative_path,
        'status': status,
    }


def _scan_relocation_candidates(
    root: Path,
    records: dict,
    missing_entries: list[tuple[str, dict]],
) -> dict[str, list[str]]:
    needed_by_size: dict[int, dict[str, str]] = {}
    for relative_path, record in missing_entries:
        digest = _valid_sha256(record.get('sha256'))
        size = record.get('file_size')
        if digest and isinstance(size, int) and not isinstance(size, bool) and size >= 0:
            needed_by_size.setdefault(size, {})[relative_path] = digest
    if not needed_by_size:
        return {}

    candidates: dict[str, list[str]] = {relative_path: [] for relative_path, _ in missing_entries}
    indexed_paths = set(records)
    for current_root, dir_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current_root)
        dir_names[:] = [
            name for name in sorted(dir_names)
            if not _is_link_or_junction(current_path / name)
        ]
        for filename in sorted(file_names):
            candidate = current_path / filename
            if _is_link_or_junction(candidate) or not is_supported_image(str(candidate)):
                continue
            try:
                file_stat = candidate.stat()
                candidate_relative = candidate.relative_to(root).as_posix()
                if candidate_relative in indexed_paths:
                    continue
                target_hashes = needed_by_size.get(file_stat.st_size)
                if not target_hashes:
                    continue
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(root)
                if not stat.S_ISREG(file_stat.st_mode):
                    continue
                digest = _file_sha256(candidate)
            except (OSError, RuntimeError, ValueError):
                continue
            for missing_path, expected_hash in target_hashes.items():
                if digest == expected_hash:
                    candidates[missing_path].append(candidate_relative)
    return {
        old_path: sorted(new_paths)
        for old_path, new_paths in candidates.items()
        if new_paths
    }


@serialize_index_update
def check_library_lifecycle(library_id: str, record_id: str | None = None) -> dict:
    index = _load_index()
    library, root = _get_library(index, library_id)
    records = library['records']
    selected_records = _select_records(records, record_id)
    checked = available = missing = modified = unverified = 0
    changed_records = []
    lifecycle_records = []
    missing_entries = []
    index_changed = False

    for relative_path, record in selected_records:
        checked += 1
        before = dict(record)
        try:
            if record.get('relative_path', relative_path) != relative_path:
                raise LifecycleError('Indexed record path does not match its index key.')
            parts = _safe_relative_parts(relative_path)
            source_candidate = root.joinpath(*parts)
            current = root
            unsafe_link = False
            for part in parts:
                current = current / part
                if _is_link_or_junction(current):
                    unsafe_link = True
                    break
            if unsafe_link:
                status_value = 'unverified'
                record['file_status'] = status_value
            elif not source_candidate.exists():
                status_value = 'missing'
                record['file_status'] = status_value
                missing_entries.append((relative_path, record))
            else:
                source = source_candidate.resolve(strict=True)
                source.relative_to(root)
                file_stat = source.stat()
                if not stat.S_ISREG(file_stat.st_mode) or not is_supported_image(str(source)):
                    status_value = 'unverified'
                    record['file_status'] = status_value
                else:
                    current_modified_time = _time_value(file_stat.st_mtime)
                    expected_hash = _valid_sha256(record.get('sha256'))
                    unchanged_stat = _stat_matches(record, file_stat.st_size, current_modified_time)
                    prior_status = record.get('file_status')

                    if unchanged_stat and prior_status not in ('modified', 'missing'):
                        status_value = 'available'
                        record['file_status'] = status_value
                        record.pop('lifecycle_observed_size', None)
                        record.pop('lifecycle_observed_modified_time', None)
                        record.pop('lifecycle_observed_sha256', None)
                    elif (
                        unchanged_stat
                        and prior_status == 'modified'
                        and _stat_matches(
                            {
                                'file_size': record.get('lifecycle_observed_size'),
                                'modified_time': record.get('lifecycle_observed_modified_time'),
                            },
                            file_stat.st_size,
                            current_modified_time,
                        )
                    ):
                        status_value = 'modified'
                    elif expected_hash:
                        current_hash = _file_sha256(source)
                        if current_hash == expected_hash:
                            status_value = 'available'
                            record['file_status'] = status_value
                            record['file_size'] = file_stat.st_size
                            record['modified_time'] = current_modified_time
                            record.pop('lifecycle_observed_size', None)
                            record.pop('lifecycle_observed_modified_time', None)
                            record.pop('lifecycle_observed_sha256', None)
                        else:
                            status_value = 'modified'
                            record['file_status'] = status_value
                            record['lifecycle_observed_size'] = file_stat.st_size
                            record['lifecycle_observed_modified_time'] = current_modified_time
                            record['lifecycle_observed_sha256'] = current_hash
                    else:
                        status_value = 'unverified'
                        record['file_status'] = status_value
        except LifecycleError:
            status_value = 'unverified'
            record['file_status'] = status_value
        except (OSError, RuntimeError, ValueError):
            status_value = 'unverified'
            record['file_status'] = status_value

        if status_value == 'available':
            available += 1
        elif status_value == 'missing':
            missing += 1
        elif status_value == 'modified':
            modified += 1
        else:
            unverified += 1

        lifecycle_records.append(_record_info(relative_path, record, status_value))
        if record != before:
            index_changed = True
            changed_records.append(_record_info(relative_path, record, status_value))
            if status_value in {'missing', 'modified', 'unverified'}:
                _invalidate_duplicate_results(library, records)

    relocation_candidates = _scan_relocation_candidates(root, records, missing_entries)
    relocation_results = []
    for relative_path, new_paths in relocation_candidates.items():
        record = records[relative_path]
        relocation_results.append({
            **_record_info(relative_path, record, 'missing'),
            'possible_relocations': new_paths,
        })
    if index_changed:
        _save_index(index)

    return {
        'success': True,
        'library_id': library_id,
        'checked': checked,
        'available': available,
        'missing': missing,
        'modified': modified,
        'unverified': unverified,
        'relocations_detected': sum(len(item['possible_relocations']) for item in relocation_results),
        'records': lifecycle_records,
        'changed_records': changed_records,
        'possible_relocations': relocation_results,
    }


@serialize_index_update
def reconcile_relocation(
    library_id: str,
    record_id: str,
    new_relative_path: str,
    confirm: bool,
) -> dict:
    if confirm is not True:
        raise LifecycleError('Explicit confirmation is required to reconcile this relocation.')
    index = _load_index()
    library, root = _get_library(index, library_id)
    records = library['records']
    selected = _select_records(records, record_id)
    if len(selected) != 1:
        raise LifecycleError('Exactly one indexed record must be selected.')
    old_relative_path, record = selected[0]

    old_candidate = root.joinpath(*_safe_relative_parts(old_relative_path))
    if old_candidate.exists() or _is_link_or_junction(old_candidate):
        raise LifecycleError('The original indexed path is still occupied; relocation cannot be applied.')

    new_path = _resolve_inside(root, new_relative_path, strict=True)
    canonical_new_path = new_path.relative_to(root).as_posix()
    if os.path.normcase(canonical_new_path.replace('\\', '/')) in {
        os.path.normcase(path.replace('\\', '/'))
        for path in records
        if isinstance(path, str)
    }:
        raise LifecycleError('The destination path is already indexed.')
    if not new_path.is_file() or not is_supported_image(str(new_path)):
        raise LifecycleError('Relocation target must be an existing supported image file.')
    file_stat = new_path.stat()
    if not stat.S_ISREG(file_stat.st_mode):
        raise LifecycleError('Relocation target must be a regular file.')
    expected_hash = _valid_sha256(record.get('sha256'))
    if expected_hash is None or _file_sha256(new_path) != expected_hash:
        raise LifecycleError('Relocation target does not match the indexed SHA-256.')

    before = dict(record)
    record['record_id'] = canonical_new_path
    record['relative_path'] = canonical_new_path
    record['filename'] = new_path.name
    record['extension'] = new_path.suffix.lower()
    record['file_size'] = file_stat.st_size
    record['modified_time'] = _time_value(file_stat.st_mtime)
    record['file_status'] = 'available'
    record['processing_status'] = 'pending'
    record.pop('lifecycle_observed_size', None)
    record.pop('lifecycle_observed_modified_time', None)
    record.pop('lifecycle_observed_sha256', None)
    records.pop(old_relative_path)
    records[canonical_new_path] = record

    _invalidate_duplicate_results(library, records)
    try:
        _save_index(index)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        records.pop(canonical_new_path, None)
        records[old_relative_path] = before
        raise RuntimeError(
            'The relocation was validated, but the local index could not be updated.'
        ) from exc

    return {
        'success': True,
        'library_id': library_id,
        'record_id': canonical_new_path,
        'previous_record_id': old_relative_path,
        'filename': new_path.name,
        'relative_path': canonical_new_path,
        'status': 'available',
        'record': {
            field: record[field]
            for field in (
                'record_id', 'filename', 'relative_path', 'extension',
                'file_size', 'modified_time', 'width', 'height',
                'aspect_ratio', 'sha256', 'phash', 'processing_status',
                'file_status',
            )
            if field in record
        },
        'message': 'Relocation reconciled. Existing AI outputs were preserved.',
    }
