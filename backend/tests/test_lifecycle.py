import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.services import (
    ai_processing_service,
    classification_service,
    dashboard_service,
    duplicate_service,
    embedding_service,
    lifecycle_service,
    library_service,
    ocr_service,
    search_service,
)

client = TestClient(app)


@pytest.fixture
def lifecycle_index(tmp_path, monkeypatch):
    index_path = tmp_path / 'data' / 'index' / 'library_index.json'
    monkeypatch.setattr(lifecycle_service, 'INDEX_PATH', index_path)
    return index_path


def create_image(path: Path, color=(30, 90, 170)) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new('RGB', (64, 48), color)
    image.save(path, format='PNG')
    return path.read_bytes()


def index_library(index_path: Path, root: Path, record_paths: list[str]):
    records = {}
    for relative_path in record_paths:
        image_path = root / relative_path
        data = image_path.read_bytes()
        records[relative_path] = {
            'record_id': relative_path,
            'filename': image_path.name,
            'relative_path': relative_path,
            'extension': image_path.suffix.lower(),
            'file_size': len(data),
            'modified_time': datetime.fromtimestamp(
                image_path.stat().st_mtime
            ).isoformat(timespec='seconds'),
            'width': 64,
            'height': 48,
            'sha256': hashlib.sha256(data).hexdigest(),
            'phash': '0123456789abcdef',
            'file_status': 'available',
            'processing_status': 'pending',
            'ai_processing_status': 'completed',
            'ai_processing_version': 'phase2-local-v1',
            'ocr_text': 'existing OCR',
            'caption': 'existing caption',
            'embedding_path': 'data/embeddings/test.npy',
            'embedding_status': 'completed',
            'classification_status': 'classified',
            'classification_version': 'phase3-multi-evidence-v1',
            'category': 'Education',
            'classification_confidence': 0.8,
            'duplicate_groups': [{'group_id': 'group', 'duplicate_type': 'exact'}],
        }
    library_id = 'library-lifecycle'
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps({
        'version': 1,
        'libraries': {
            library_id: {
                'library_root': str(root.resolve()),
                'records': records,
                'duplicate_detection': {'groups': []},
            },
        },
    }), encoding='utf-8')
    return library_id, records


def load_index(index_path: Path):
    return json.loads(index_path.read_text(encoding='utf-8'))


def check(library_id: str, record_id: str | None = None):
    body = {'library_id': library_id}
    if record_id is not None:
        body['record_id'] = record_id
    return client.post('/lifecycle/check', json=body)


def test_unchanged_file_remains_available_without_rewriting_index(
    lifecycle_index, tmp_path, monkeypatch,
):
    root = tmp_path / 'library'
    original = create_image(root / 'same.png')
    library_id, records = index_library(lifecycle_index, root, ['same.png'])
    before_index = lifecycle_index.read_bytes()
    before_record = records['same.png'].copy()

    def unexpected(*_args, **_kwargs):
        pytest.fail('Lifecycle check invoked an AI or duplicate processing service')

    monkeypatch.setattr(ocr_service, 'run_ocr', unexpected)
    monkeypatch.setattr(ai_processing_service, 'generate_caption', unexpected)
    monkeypatch.setattr(embedding_service, 'generate_embedding', unexpected)
    monkeypatch.setattr(classification_service, 'process_library_classification', unexpected)
    monkeypatch.setattr(duplicate_service, 'process_library_duplicates', unexpected)

    response = check(library_id)

    assert response.status_code == 200
    result = response.json()
    assert result['checked'] == result['available'] == 1
    assert result['changed_records'] == []
    assert result['relocations_detected'] == 0
    assert lifecycle_index.read_bytes() == before_index
    saved = load_index(lifecycle_index)['libraries'][library_id]['records']['same.png']
    for key in (
        'sha256', 'phash', 'ocr_text', 'caption', 'embedding_path',
        'classification_version', 'ai_processing_version',
    ):
        assert saved[key] == before_record[key]
    assert (root / 'same.png').read_bytes() == original


def test_missing_and_reappearing_identical_file_transitions(lifecycle_index, tmp_path):
    root = tmp_path / 'library'
    original = create_image(root / 'same.png')
    library_id, _ = index_library(lifecycle_index, root, ['same.png'])
    (root / 'same.png').unlink()

    missing_response = check(library_id)

    assert missing_response.status_code == 200
    assert missing_response.json()['missing'] == 1
    saved_library = load_index(lifecycle_index)['libraries'][library_id]
    saved_after_missing = saved_library['records']['same.png']
    assert saved_after_missing['file_status'] == 'missing'
    assert 'same.png' in saved_library['records']
    assert 'duplicate_groups' not in saved_after_missing
    assert 'duplicate_detection' not in saved_library
    assert not (root / 'same.png').exists()

    (root / 'same.png').write_bytes(original)
    restored = check(library_id)
    assert restored.status_code == 200
    assert restored.json()['available'] == 1
    saved_after_restore = load_index(lifecycle_index)['libraries'][library_id]['records']['same.png']
    assert saved_after_restore['file_status'] == 'available'


def test_external_change_is_marked_modified_and_ai_data_is_preserved(
    lifecycle_index, tmp_path, monkeypatch,
):
    root = tmp_path / 'library'
    original = create_image(root / 'changed.png')
    library_id, records = index_library(lifecycle_index, root, ['changed.png'])
    old_hash = records['changed.png']['sha256']
    changed = create_image(root / 'changed.png', color=(200, 20, 40))
    assert changed != original
    new_modified_time = root.joinpath('changed.png').stat().st_mtime + 5
    os.utime(root / 'changed.png', (new_modified_time, new_modified_time))
    monkeypatch.setattr(
        ai_processing_service,
        'process_library_images',
        lambda *_args, **_kwargs: pytest.fail('Lifecycle automatically reran AI processing'),
    )

    response = check(library_id)

    assert response.status_code == 200
    assert response.json()['modified'] == 1
    saved = load_index(lifecycle_index)['libraries'][library_id]['records']['changed.png']
    assert saved['file_status'] == 'modified'
    assert saved['sha256'] == old_hash
    assert saved['ai_processing_status'] == 'completed'
    assert saved['lifecycle_observed_sha256'] == hashlib.sha256(changed).hexdigest()
    assert saved['ocr_text'] == 'existing OCR'
    assert saved['caption'] == 'existing caption'
    assert saved['embedding_path'] == 'data/embeddings/test.npy'
    assert saved['classification_status'] == 'classified'


def test_reappearing_different_content_is_modified(lifecycle_index, tmp_path):
    root = tmp_path / 'library'
    original = create_image(root / 'restore.png')
    library_id, _ = index_library(lifecycle_index, root, ['restore.png'])
    (root / 'restore.png').unlink()
    assert check(library_id).json()['missing'] == 1
    create_image(root / 'restore.png', color=(5, 220, 55))

    response = check(library_id)

    assert response.status_code == 200
    assert response.json()['modified'] == 1
    assert load_index(lifecycle_index)['libraries'][library_id]['records']['restore.png']['file_status'] == 'modified'
    assert hashlib.sha256((root / 'restore.png').read_bytes()).hexdigest() != hashlib.sha256(original).hexdigest()


def test_exact_relocation_is_detected_but_not_applied_until_confirmed(
    lifecycle_index, tmp_path,
):
    root = tmp_path / 'library'
    original = create_image(root / 'old-name.png')
    library_id, records = index_library(lifecycle_index, root, ['old-name.png'])
    previous_hash = records['old-name.png']['sha256']
    (root / 'old-name.png').rename(root / 'new-name.png')

    detected = check(library_id)

    assert detected.status_code == 200
    result = detected.json()
    assert result['missing'] == 1
    assert result['relocations_detected'] == 1
    relocation = result['possible_relocations'][0]
    assert relocation['relative_path'] == 'old-name.png'
    assert relocation['possible_relocations'] == ['new-name.png']
    saved = load_index(lifecycle_index)['libraries'][library_id]
    assert 'old-name.png' in saved['records']
    assert 'new-name.png' not in saved['records']
    assert (root / 'new-name.png').read_bytes() == original

    rejected = client.post('/lifecycle/reconcile', json={
        'library_id': library_id,
        'record_id': 'old-name.png',
        'new_relative_path': 'new-name.png',
        'confirm': False,
    })
    assert rejected.status_code == 400
    assert 'old-name.png' in load_index(lifecycle_index)['libraries'][library_id]['records']

    reconciled = client.post('/lifecycle/reconcile', json={
        'library_id': library_id,
        'record_id': 'old-name.png',
        'new_relative_path': 'new-name.png',
        'confirm': True,
    })
    assert reconciled.status_code == 200
    assert reconciled.json()['previous_record_id'] == 'old-name.png'
    saved_record = load_index(lifecycle_index)['libraries'][library_id]['records']['new-name.png']
    assert saved_record['record_id'] == saved_record['relative_path'] == 'new-name.png'
    assert saved_record['sha256'] == previous_hash
    assert saved_record['file_status'] == 'available'
    assert saved_record['ai_processing_version'] == 'phase2-local-v1'
    assert saved_record['ocr_text'] == 'existing OCR'
    assert saved_record['embedding_path'] == 'data/embeddings/test.npy'
    assert saved_record['classification_status'] == 'classified'
    assert 'duplicate_detection' not in load_index(lifecycle_index)['libraries'][library_id]
    assert 'duplicate_groups' not in saved_record


@pytest.mark.parametrize('path', [
    '../outside.png',
    '..\\outside.png',
    'C:\\outside.png',
    '\\\\server\\share\\outside.png',
    'folder/../../escape.png',
])
def test_relocation_candidates_outside_library_are_rejected(
    lifecycle_index, tmp_path, path,
):
    root = tmp_path / 'library'
    create_image(root / 'old.png')
    library_id, _ = index_library(lifecycle_index, root, ['old.png'])
    (root / 'old.png').unlink()

    response = client.post('/lifecycle/reconcile', json={
        'library_id': library_id,
        'record_id': 'old.png',
        'new_relative_path': path,
        'confirm': True,
    })

    assert response.status_code == 400
    assert 'old.png' in load_index(lifecycle_index)['libraries'][library_id]['records']


def test_invalid_library_and_record_ids_return_clear_client_errors(lifecycle_index, tmp_path):
    root = tmp_path / 'library'
    create_image(root / 'one.png')
    library_id, _ = index_library(lifecycle_index, root, ['one.png'])

    unknown_library = check('not-a-library')
    unknown_record = check(library_id, 'not-a-record')

    assert unknown_library.status_code == 400
    assert unknown_record.status_code == 400


def test_missing_hash_metadata_is_unverified_and_bad_index_path_is_not_missing(
    lifecycle_index, tmp_path,
):
    root = tmp_path / 'library'
    create_image(root / 'one.png')
    library_id, _ = index_library(lifecycle_index, root, ['one.png'])
    payload = load_index(lifecycle_index)
    record = payload['libraries'][library_id]['records']['one.png']
    record.pop('sha256')
    record.pop('file_size')
    record.pop('modified_time')
    payload['libraries'][library_id]['records']['../escape.png'] = {
        'record_id': '../escape.png',
        'relative_path': '../escape.png',
        'file_status': 'available',
    }
    lifecycle_index.write_text(json.dumps(payload), encoding='utf-8')

    response = check(library_id)

    assert response.status_code == 200
    result = response.json()
    assert result['unverified'] == 2
    assert result['missing'] == 0
    assert load_index(lifecycle_index)['libraries'][library_id]['records']['../escape.png']['file_status'] == 'unverified'


def test_multiple_libraries_are_isolated_and_library_filter_is_explicit(lifecycle_index, tmp_path):
    root_a = tmp_path / 'a'
    root_b = tmp_path / 'b'
    create_image(root_a / 'a.png')
    create_image(root_b / 'b.png')
    library_id, _ = index_library(lifecycle_index, root_a, ['a.png'])
    payload = load_index(lifecycle_index)
    payload['libraries']['other-library'] = {
        'library_root': str(root_b.resolve()),
        'records': {
            'b.png': {
                'record_id': 'b.png',
                'relative_path': 'b.png',
                'file_status': 'missing',
            },
        },
    }
    lifecycle_index.write_text(json.dumps(payload), encoding='utf-8')

    response = check(library_id)

    assert response.status_code == 200
    assert response.json()['checked'] == 1
    assert response.json()['available'] == 1
    saved = load_index(lifecycle_index)
    assert saved['libraries']['other-library']['records']['b.png']['file_status'] == 'missing'


def test_lifecycle_does_not_copy_or_delete_source_and_persisted_json_stays_valid(
    lifecycle_index, tmp_path,
):
    root = tmp_path / 'library'
    image_bytes = create_image(root / 'one.png')
    library_id, _ = index_library(lifecycle_index, root, ['one.png'])

    response = check(library_id)

    assert response.status_code == 200
    assert (root / 'one.png').read_bytes() == image_bytes
    assert list(root.rglob('*.png')) == [root / 'one.png']
    assert isinstance(load_index(lifecycle_index), dict)


def test_changed_records_are_not_included_in_semantic_search(
    lifecycle_index, tmp_path, monkeypatch,
):
    root = tmp_path / 'library'
    create_image(root / 'changed.png')
    library_id, _ = index_library(lifecycle_index, root, ['changed.png'])
    payload = load_index(lifecycle_index)
    payload['libraries'][library_id]['records']['changed.png']['file_status'] = 'modified'
    lifecycle_index.write_text(json.dumps(payload), encoding='utf-8')
    monkeypatch.setattr(search_service, 'INDEX_PATH', lifecycle_index)
    monkeypatch.setattr(
        search_service,
        'encode_text_embeddings',
        lambda _texts: [[1.0, 0.0]],
    )
    monkeypatch.setattr(
        search_service,
        '_record_embedding',
        lambda _record: pytest.fail('Modified record embedding was loaded for search'),
    )

    response = client.post('/search', json={
        'folder_path': str(root),
        'query': 'changed image',
    })

    assert response.status_code == 200
    assert response.json()['total_results'] == 0


def test_phase1_scan_external_change_move_reconcile_and_dashboard_integration(
    lifecycle_index, tmp_path, monkeypatch,
):
    root = tmp_path / 'library'
    create_image(root / 'first.png', color=(12, 34, 56))
    second_bytes = create_image(root / 'second.png', color=(180, 40, 20))
    monkeypatch.setattr(library_service, 'INDEX_PATH', lifecycle_index)
    monkeypatch.setattr(library_service, 'INDEX_DIR', lifecycle_index.parent)
    monkeypatch.setattr(dashboard_service, 'INDEX_PATH', lifecycle_index)

    scan = library_service.scan_library(str(root))
    library_id = scan['library_id']
    initial = check(library_id)
    assert initial.status_code == 200
    assert initial.json()['available'] == 2
    index_snapshot = lifecycle_index.read_bytes()
    assert check(library_id).status_code == 200
    assert lifecycle_index.read_bytes() == index_snapshot

    create_image(root / 'first.png', color=(1, 250, 3))
    changed_time = (root / 'first.png').stat().st_mtime + 4
    os.utime(root / 'first.png', (changed_time, changed_time))
    (root / 'relocated').mkdir()
    (root / 'second.png').rename(root / 'relocated' / 'second.png')

    checked = check(library_id)
    assert checked.status_code == 200
    result = checked.json()
    assert result['modified'] == 1
    assert result['missing'] == 1
    assert result['relocations_detected'] == 1
    assert {record['status'] for record in result['records']} == {'modified', 'missing'}
    assert load_index(lifecycle_index)['libraries'][library_id]['records']['second.png']['file_status'] == 'missing'

    relocation = result['possible_relocations'][0]
    reconciled = client.post('/lifecycle/reconcile', json={
        'library_id': library_id,
        'record_id': relocation['record_id'],
        'new_relative_path': 'relocated/second.png',
        'confirm': True,
    })
    assert reconciled.status_code == 200
    saved = load_index(lifecycle_index)['libraries'][library_id]
    assert 'second.png' not in saved['records']
    assert saved['records']['relocated/second.png']['sha256'] == hashlib.sha256(second_bytes).hexdigest()
    assert saved['records']['relocated/second.png']['file_status'] == 'available'
    assert (root / 'relocated' / 'second.png').read_bytes() == second_bytes

    dashboard = client.get('/dashboard/stats', params={'folder_path': str(root)})
    assert dashboard.status_code == 200
    stats = dashboard.json()
    assert stats['total_images'] == 2
    assert stats['file_status']['modified'] == 1
    assert stats['file_status']['available'] == 1
    assert stats['file_status']['missing'] == 0
    assert (root / 'first.png').exists()
    assert not (root / 'second.png').exists()
    assert list(root.rglob('*.png')) == [root / 'first.png', root / 'relocated' / 'second.png']
