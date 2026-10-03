import json
from datetime import datetime
from pathlib import Path

from app.services.index_storage import save_json_atomically, serialize_index_update
from app.services.library_service import INDEX_PATH

IMPORTANCE_STATUSES = ('critical', 'important', 'normal', 'low')
PROTECTION_STATUSES = ('protected', 'unprotected')
LIFECYCLE_STATUSES = ('keep', 'review', 'archive', 'delete')


class RecordStateError(ValueError):
    pass


def _save_index(payload: dict) -> None:
    save_json_atomically(INDEX_PATH, payload)


def _load_index() -> dict:
    if not INDEX_PATH.exists():
        raise RecordStateError('The local library index does not exist.')
    try:
        with INDEX_PATH.open('r', encoding='utf-8') as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RecordStateError('The local library index could not be read.') from exc
    if not isinstance(payload, dict) or not isinstance(payload.get('libraries'), dict):
        raise RecordStateError('The local library index has an invalid structure.')
    return payload


def _find_record(index: dict, library_id: str, record_id: str) -> tuple[dict, str, dict]:
    library = index['libraries'].get(library_id)
    if not isinstance(library, dict) or not isinstance(library.get('records'), dict):
        raise RecordStateError('Indexed library was not found.')
    records = library['records']
    relative_path = record_id if record_id in records else next(
        (path for path, record in records.items() if isinstance(record, dict) and record.get('record_id') == record_id),
        None,
    )
    if relative_path is None or not isinstance(records[relative_path], dict):
        raise RecordStateError('Indexed image record was not found.')
    return library, relative_path, records[relative_path]


@serialize_index_update
def update_record_state(
    library_id: str,
    record_id: str,
    *,
    importance_status: str | None = None,
        is_important: bool | None = None,
    protection_status: str | None = None,
    lifecycle_status: str | None = None,
) -> dict:
    values = {
        'importance_status': (importance_status, IMPORTANCE_STATUSES),
        'protection_status': (protection_status, PROTECTION_STATUSES),
        'lifecycle_status': (lifecycle_status, LIFECYCLE_STATUSES),
    }
        if is_important is not None and not isinstance(is_important, bool):
            raise RecordStateError('is_important must be a boolean.')
    for field, (value, allowed) in values.items():
        if value is not None and value not in allowed:
            raise RecordStateError(f'Invalid {field.replace("_", " ")}.')
    index = _load_index()
    _library, relative_path, record = _find_record(index, library_id, record_id)
    for field, (value, _allowed) in values.items():
        if value is not None:
            record[field] = value
            record[f'{field.split("_")[0]}_source'] = 'manual'
        if is_important is not None:
            record['is_important'] = is_important
            record['importance_marked_at'] = datetime.now().astimezone().isoformat(timespec='seconds') if is_important else None
            record['importance_source'] = 'manual'
            record['importance_status'] = 'important' if is_important else 'normal'
    record['record_state_updated_at'] = datetime.now().astimezone().isoformat(timespec='seconds')
    _save_index(index)
    return {
        'library_id': library_id,
        'record_id': record.get('record_id') or relative_path,
        'relative_path': relative_path,
        'record': record,
    }