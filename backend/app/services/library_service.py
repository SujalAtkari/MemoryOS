import hashlib
import json
import os
from pathlib import Path

from app.services.hashing_service import is_supported_image
from app.services.index_storage import save_json_atomically, serialize_index_update
from app.services.metadata_service import extract_image_metadata

INDEX_DIR = Path(__file__).resolve().parents[3] / 'data' / 'index'
INDEX_PATH = INDEX_DIR / 'library_index.json'

_PRESERVED_RECORD_FIELDS = (
    'category',
    'subcategory',
    'classification_source',
    'manual_classification',
    'previous_category',
    'importance_score',
    'importance_status',
        'is_important',
        'importance_marked_at',
    'importance_source',
    'protection_status',
    'protection_source',
    'lifecycle_status',
)


def _library_id_for_path(folder_path: str) -> str:
    normalized = os.path.normpath(folder_path)
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:12]


def _load_index() -> dict:
    if not INDEX_PATH.exists():
        return {'version': 1, 'libraries': {}}

    try:
        with INDEX_PATH.open('r', encoding='utf-8') as file_handle:
            payload = json.load(file_handle)
        if isinstance(payload, dict) and 'libraries' in payload:
            return payload
    except (OSError, json.JSONDecodeError):
        pass

    return {'version': 1, 'libraries': {}}


def _save_index(payload: dict) -> None:
    save_json_atomically(INDEX_PATH, payload)


def validate_folder_path(folder_path: str) -> str:
    if not folder_path or not str(folder_path).strip():
        raise ValueError('Folder path is required.')

    normalized = Path(folder_path).expanduser()
    if not normalized.exists():
        raise FileNotFoundError('Folder does not exist.')
    if not normalized.is_dir():
        raise ValueError('Path is not a directory.')

    try:
        resolved = normalized.resolve()
    except OSError as exc:
        raise ValueError('Folder path could not be resolved.') from exc

    return str(resolved)


@serialize_index_update
def scan_library(folder_path: str) -> dict:
    library_root = validate_folder_path(folder_path)
    library_id = _library_id_for_path(library_root)
    index = _load_index()
    library_index = index.setdefault('libraries', {}).get(library_id)

    if library_index is None:
        library_index = {'library_root': library_root, 'records': {}}
        index['libraries'][library_id] = library_index

    library_index['library_root'] = library_root
    discovered_paths = set()
    new_images = 0
    updated_images = 0
    unchanged_images = 0
    failed_files = []

    for root, dir_names, file_names in os.walk(library_root):
        dir_names.sort()
        for file_name in sorted(file_names):
            file_path = os.path.join(root, file_name)
            if not is_supported_image(file_path):
                continue

            relative_path = os.path.relpath(file_path, library_root).replace('\\', '/')
            discovered_paths.add(relative_path)

            try:
                metadata = extract_image_metadata(file_path, relative_path)
                existing = library_index['records'].get(relative_path)

                if existing is None:
                    record = {
                        'record_id': relative_path,
                        **metadata,
                        'file_status': 'available',
                        'processing_status': 'pending',
                        'classification_status': 'pending',
                        'importance_status': 'normal',
                        'protection_status': 'unprotected',
                        'lifecycle_status': 'keep',
                    }
                    library_index['records'][relative_path] = record
                    new_images += 1
                    continue

                changed = (
                    existing.get('sha256') != metadata['sha256']
                    or existing.get('phash') != metadata['phash']
                    or existing.get('file_size') != metadata['file_size']
                    or existing.get('modified_time') != metadata['modified_time']
                    or existing.get('width') != metadata['width']
                    or existing.get('height') != metadata['height']
                )

                if changed:
                    record = {
                        'record_id': relative_path,
                        **metadata,
                        'file_status': 'modified',
                        'processing_status': 'pending',
                    }
                    for field in _PRESERVED_RECORD_FIELDS:
                        if field in existing:
                            record[field] = existing[field]
                    library_index['records'][relative_path] = record
                    updated_images += 1
                else:
                    record = dict(existing)
                    record['file_status'] = 'available'
                    record['processing_status'] = 'pending'
                    library_index['records'][relative_path] = record
                    unchanged_images += 1

            except (PermissionError, OSError, ValueError) as exc:
                failed_files.append({'relative_path': relative_path, 'error': str(exc)})

    for relative_path, record in list(library_index['records'].items()):
        if relative_path not in discovered_paths:
            record = dict(record)
            record['file_status'] = 'missing'
            record['processing_status'] = 'pending'
            library_index['records'][relative_path] = record

    _save_index(index)

    records = sorted(library_index['records'].values(), key=lambda item: item['relative_path'])
    return {
        'library_id': library_id,
        'library_root': library_root,
        'total_discovered_images': len(discovered_paths),
        'new_images': new_images,
        'updated_images': updated_images,
        'unchanged_images': unchanged_images,
        'failed_files': failed_files,
        'image_records': records,
    }
