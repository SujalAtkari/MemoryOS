import hashlib
import json
import os
from pathlib import Path

from app.services.caption_service import generate_caption
from app.services.embedding_service import compute_embedding_path, generate_embedding
from app.services.index_storage import save_json_atomically, serialize_index_update
from app.services.library_service import INDEX_PATH, validate_folder_path
from app.services.ocr_service import run_ocr

AI_PROCESSING_VERSION = 'phase2-local-v1'


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _record_should_skip(record: dict, profile: str) -> bool:
    if record.get('ai_processing_version') != AI_PROCESSING_VERSION:
        return False
    if record.get('ai_processing_status') != 'completed':
        return False
    if not record.get('sha256'):
        return False
    embedding_path = record.get('embedding_path')
    if not embedding_path or not Path(embedding_path).exists():
        return False
    if profile == 'fast':
        return True
    if record.get('ai_processing_profile') == 'detailed':
        return True
    return bool(
        record.get('ocr_text') is not None
        and record.get('caption') is not None
        and record.get('ocr_status') not in (None, 'not_run')
        and record.get('caption_status') not in (None, 'not_run')
    )


def _load_index() -> dict:
    if not INDEX_PATH.exists():
        return {'version': 1, 'libraries': {}}
    try:
        with INDEX_PATH.open('r', encoding='utf-8') as handle:
            payload = json.load(handle)
        if isinstance(payload, dict) and 'libraries' in payload:
            return payload
    except (OSError, json.JSONDecodeError):
        pass
    return {'version': 1, 'libraries': {}}


def _save_index(payload: dict) -> None:
    save_json_atomically(INDEX_PATH, payload)


def _get_library_record(library_root: str, relative_path: str, index: dict) -> dict | None:
    library_id = hashlib.sha256(os.path.normpath(library_root).encode('utf-8')).hexdigest()[:12]
    library_index = index.get('libraries', {}).get(library_id)
    if library_index is None:
        return None
    return library_index.get('records', {}).get(relative_path)


@serialize_index_update
def process_library_images(
    folder_path: str,
    force: bool = False,
    record_ids: list[str] | None = None,
    profile: str = 'detailed',
) -> dict:
    if profile not in {'fast', 'detailed'}:
        raise ValueError('Processing profile must be fast or detailed.')

    library_root = validate_folder_path(folder_path)
    index = _load_index()
    library_id = hashlib.sha256(os.path.normpath(library_root).encode('utf-8')).hexdigest()[:12]
    library_index = index.setdefault('libraries', {}).setdefault(library_id, {'library_root': library_root, 'records': {}})
    library_index['library_root'] = library_root

    records = library_index.get('records', {})
    selected_ids = set(record_ids or [])

    processed = 0
    skipped = 0
    failed = 0
    errors = []
    image_statuses = []

    for relative_path, record in list(records.items()):
        if selected_ids and relative_path not in selected_ids and record.get('record_id') not in selected_ids:
            continue

        if not force and _record_should_skip(record, profile):
            skipped += 1
            image_statuses.append({
                'record_id': record.get('record_id') or relative_path,
                'relative_path': relative_path,
                'ai_processing_status': record.get('ai_processing_status'),
                'ai_processing_profile': record.get('ai_processing_profile', 'detailed'),
                'ocr_status': record.get('ocr_status'),
                'caption_status': record.get('caption_status'),
                'embedding_status': record.get('embedding_status'),
            })
            continue

        image_path = str(Path(library_root) / relative_path)
        if not Path(image_path).exists():
            record['ai_processing_status'] = 'failed'
            record['ai_processing_error'] = 'Image file missing from disk.'
            failed += 1
            errors.append({'relative_path': relative_path, 'error': record['ai_processing_error']})
            image_statuses.append({
                'record_id': record.get('record_id') or relative_path,
                'relative_path': relative_path,
                'ai_processing_status': record['ai_processing_status'],
                'ocr_status': record.get('ocr_status'),
                'caption_status': record.get('caption_status'),
                'embedding_status': record.get('embedding_status'),
            })
            continue

        record['ai_processing_status'] = 'processing'
        record['ai_processing_version'] = AI_PROCESSING_VERSION
        record['ai_processing_profile'] = profile
        record['ocr_status'] = 'not_run' if profile == 'fast' else 'pending'
        record['caption_status'] = 'not_run' if profile == 'fast' else 'pending'
        record['embedding_status'] = 'pending'
        record['ai_processing_error'] = None

        if profile == 'detailed':
            ocr_result = run_ocr(image_path)
            ocr_text = ocr_result.get('text')
            if ocr_text:
                record['ocr_text'] = ocr_text
                record['ocr_confidence'] = ocr_result.get('confidence')
            elif record.get('ocr_text'):
                ocr_result['status'] = 'completed'
            record['ocr_status'] = ocr_result.get('status', 'failed')
            if ocr_result.get('error'):
                record['ai_processing_error'] = ocr_result['error']

            caption_result = generate_caption(image_path)
            caption = caption_result.get('caption')
            if caption:
                record['caption'] = caption
            elif record.get('caption'):
                caption_result['status'] = 'completed'
            record['caption_status'] = caption_result.get('status', 'failed')
            if caption_result.get('error'):
                record['ai_processing_error'] = (record.get('ai_processing_error') or '') + ('; ' if record.get('ai_processing_error') else '') + caption_result['error']

        embedding_result = generate_embedding(image_path, record.get('record_id') or relative_path, str(_project_root()))
        embedding_path = embedding_result.get('embedding_path')
        if embedding_path:
            record['embedding_path'] = embedding_path
        elif record.get('embedding_path') and Path(record['embedding_path']).exists():
            embedding_result['embedding_status'] = 'completed'
        record['embedding_status'] = embedding_result.get('embedding_status', 'failed')
        if embedding_result.get('error'):
            record['ai_processing_error'] = (record.get('ai_processing_error') or '') + ('; ' if record.get('ai_processing_error') else '') + embedding_result['error']

        has_failures = record.get('embedding_status') == 'failed' or (
            profile == 'detailed'
            and any([
                record.get('ocr_status') == 'failed',
                record.get('caption_status') == 'failed',
            ])
        )

        if has_failures:
            record['ai_processing_status'] = 'failed'
            failed += 1
        else:
            record['ai_processing_status'] = 'completed'
            processed += 1

        if record.get('ai_processing_error'):
            errors.append({'relative_path': relative_path, 'error': record['ai_processing_error']})

        records[relative_path] = record
        image_statuses.append({
            'record_id': record.get('record_id') or relative_path,
            'relative_path': relative_path,
            'ai_processing_status': record['ai_processing_status'],
            'ai_processing_profile': record['ai_processing_profile'],
            'ocr_status': record.get('ocr_status'),
            'caption_status': record.get('caption_status'),
            'embedding_status': record.get('embedding_status'),
        })

    _save_index(index)

    return {
        'library_id': library_id,
        'library_root': library_root,
        'total_images': len(records),
        'processed': processed,
        'skipped': skipped,
        'failed': failed,
        'processing_status': 'completed' if failed == 0 else 'partial_failure',
        'errors': errors,
        'image_statuses': image_statuses,
        'version': AI_PROCESSING_VERSION,
        'profile': profile,
    }


@serialize_index_update
def process_library_captions(
    folder_path: str,
    record_ids: list[str] | None = None,
    force: bool = False,
) -> dict:
    library_root = validate_folder_path(folder_path)
    index = _load_index()
    library_id = hashlib.sha256(os.path.normpath(library_root).encode('utf-8')).hexdigest()[:12]
    library_index = index.get('libraries', {}).get(library_id)
    if library_index is None:
        raise ValueError('Library has not been scanned yet.')

    records = library_index.get('records', {})
    selected_ids = set(record_ids or [])
    processed = skipped = failed = 0
    errors = []
    image_statuses = []
    captions_by_sha256 = {
        record.get('sha256'): str(record.get('caption') or '').strip()
        for library in index.get('libraries', {}).values()
        for record in library.get('records', {}).values()
        if record.get('sha256')
        and record.get('caption_status') == 'completed'
        and str(record.get('caption') or '').strip()
    }

    for relative_path, record in list(records.items()):
        if selected_ids and relative_path not in selected_ids and record.get('record_id') not in selected_ids:
            continue
        if record.get('file_status') in {'missing', 'modified', 'unverified'}:
            skipped += 1
            continue
        if (
            not force
            and record.get('caption_status') == 'completed'
            and str(record.get('caption') or '').strip()
        ):
            skipped += 1
            image_statuses.append({
                'record_id': record.get('record_id') or relative_path,
                'relative_path': relative_path,
                'caption_status': 'completed',
                'caption': record['caption'],
                'caption_error': None,
            })
            continue

        cached_caption = captions_by_sha256.get(record.get('sha256'))
        if not force and cached_caption:
            record['caption'] = cached_caption
            record['caption_status'] = 'completed'
            record['caption_error'] = None
            processed += 1
            image_statuses.append({
                'record_id': record.get('record_id') or relative_path,
                'relative_path': relative_path,
                'caption_status': 'completed',
                'caption': cached_caption,
                'caption_error': None,
            })
            records[relative_path] = record
            continue

        image_path = Path(library_root) / relative_path
        if not image_path.is_file():
            error = 'Image file missing from disk.'
            record['caption_status'] = 'failed'
            record['caption_error'] = error
            failed += 1
            errors.append({'relative_path': relative_path, 'error': error})
        else:
            result = generate_caption(str(image_path))
            caption = str(result.get('caption') or '').strip()
            if caption:
                record['caption'] = caption
                record['caption_status'] = 'completed'
                record['caption_error'] = None
                if record.get('sha256'):
                    captions_by_sha256[record['sha256']] = caption
                processed += 1
            else:
                record['caption_status'] = 'failed'
                failed += 1
                error = str(result.get('error') or 'Caption model returned no caption.')
                record['caption_error'] = error
                errors.append({'relative_path': relative_path, 'error': error})

        image_statuses.append({
            'record_id': record.get('record_id') or relative_path,
            'relative_path': relative_path,
            'caption_status': record['caption_status'],
            'caption': record.get('caption') or '',
            'caption_error': record.get('caption_error'),
        })
        records[relative_path] = record

    _save_index(index)
    return {
        'library_id': library_id,
        'library_root': library_root,
        'total': processed + skipped + failed,
        'processed': processed,
        'skipped': skipped,
        'failed': failed,
        'processing_status': 'completed' if failed == 0 else 'partial_failure',
        'errors': errors,
        'image_statuses': image_statuses,
    }
