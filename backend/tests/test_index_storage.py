import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.services import index_storage
from app.services.index_storage import save_json_atomically, serialize_index_update


def test_concurrent_index_updates_are_serialized_and_atomic(tmp_path):
    index_path = tmp_path / 'library_index.json'
    index_path.write_text(json.dumps({'count': 0}), encoding='utf-8')

    @serialize_index_update
    def increment_index():
        payload = json.loads(index_path.read_text(encoding='utf-8'))
        payload['count'] += 1
        save_json_atomically(index_path, payload)

    with ThreadPoolExecutor(max_workers=12) as executor:
        list(executor.map(lambda _: increment_index(), range(60)))

    assert json.loads(index_path.read_text(encoding='utf-8')) == {'count': 60}
    assert list(tmp_path.glob('.library_index.json.*.tmp')) == []


def test_atomic_index_save_retries_windows_sharing_violation(tmp_path, monkeypatch):
    index_path = tmp_path / 'library_index.json'
    real_replace = os.replace
    attempts = 0

    def fail_twice_with_sharing_violation(source: Path, destination: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            error = PermissionError(32, 'The process cannot access the file')
            error.winerror = 32
            raise error
        real_replace(source, destination)

    monkeypatch.setattr(index_storage.os, 'replace', fail_twice_with_sharing_violation)

    save_json_atomically(index_path, {'status': 'saved'})

    assert attempts == 3
    assert json.loads(index_path.read_text(encoding='utf-8')) == {'status': 'saved'}
    assert list(tmp_path.glob('.library_index.json.*.tmp')) == []
