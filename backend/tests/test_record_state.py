import json

from fastapi.testclient import TestClient

from app.main import app
from app.services import record_state_service

client = TestClient(app)


def test_record_state_updates_are_persisted(tmp_path, monkeypatch):
    index_path = tmp_path / 'library_index.json'
    monkeypatch.setattr(record_state_service, 'INDEX_PATH', index_path)
    index_path.write_text(json.dumps({
        'version': 1,
        'libraries': {
            'library': {
                'library_root': str(tmp_path),
                'records': {
                    'photo.jpg': {
                        'record_id': 'photo.jpg',
                        'relative_path': 'photo.jpg',
                        'category': 'Others',
                    },
                },
            },
        },
    }), encoding='utf-8')

    response = client.patch('/records/state', json={
        'library_id': 'library',
        'record_id': 'photo.jpg',
        'is_important': True,
        'importance_status': 'important',
        'protection_status': 'protected',
        'lifecycle_status': 'review',
    })

    assert response.status_code == 200
    record = response.json()['record']
    assert record['importance_status'] == 'important'
    assert record['is_important'] is True
    assert record['importance_source'] == 'manual'
    assert record['importance_marked_at']
    assert record['protection_status'] == 'protected'
    assert record['lifecycle_status'] == 'review'
    assert record['importance_source'] == 'manual'
    assert json.loads(index_path.read_text(encoding='utf-8'))['libraries']['library']['records']['photo.jpg']['protection_status'] == 'protected'