import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import ai_processing_service, dashboard_service, duplicate_service

client = TestClient(app)


@pytest.fixture
def dashboard_index(tmp_path, monkeypatch):
    index_path = tmp_path / 'data' / 'index' / 'library_index.json'
    monkeypatch.setattr(dashboard_service, 'INDEX_PATH', index_path)
    return index_path


def write_index(path: Path, libraries: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'version': 1, 'libraries': libraries}), encoding='utf-8')


def test_missing_index_returns_empty_dashboard(dashboard_index):
    response = client.get('/dashboard/stats')

    assert response.status_code == 200
    stats = response.json()
    assert stats['total_images'] == 0
    assert stats['library_count'] == 0
    assert stats['processing']['pending'] == 0
    assert stats['categories']['distribution'][0]['percentage'] == 0
    assert stats['duplicates']['groups'] == 0


def test_dashboard_aggregates_status_categories_duplicates_and_libraries(dashboard_index, tmp_path):
    first_root = tmp_path / 'family_photos'
    second_root = tmp_path / 'work'
    first_records = {
        'one.jpg': {
            'ai_processing_status': 'completed',
            'classification_status': 'classified',
            'category': 'Government & Identity',
            'file_status': 'available',
            'duplicate_groups': [
                {'group_id': 'shared', 'duplicate_type': 'exact'},
                {'group_id': 'visual-a', 'duplicate_type': 'visual'},
            ],
        },
        'two.jpg': {
            'ai_processing_status': 'failed',
            'classification_status': 'needs_review',
            'category': None,
            'file_status': 'missing',
            'duplicate_groups': [
                {'group_id': 'shared', 'duplicate_type': 'exact'},
                {'group_id': 'visual-a', 'duplicate_type': 'visual'},
            ],
        },
        'three.jpg': {
            'ai_processing_status': 'pending',
            'classification_status': 'failed',
            'category': None,
            'file_status': 'modified',
            'duplicate_groups': [],
        },
        'four.jpg': {
            'ai_processing_status': 'processing',
            'classification_status': 'classified',
            'category': 'Education',
            'file_status': 'available',
            'duplicate_groups': [],
        },
        'five.jpg': {
            'classification_status': 'classified',
            'category': 'Government & Identity',
        },
    }
    second_records = {
        'report.png': {
            'ai_processing_status': 'completed',
            'classification_status': 'classified',
            'category': 'Education',
            'file_status': 'available',
        },
        'bad-record': None,
        'legacy.jpg': {},
    }
    write_index(dashboard_index, {
        'library-a': {
            'library_root': str(first_root),
            'records': first_records,
            'duplicate_detection': {
                'groups': [
                    {
                        'group_id': 'shared',
                        'duplicate_type': 'exact',
                        'members': [{'relative_path': 'one.jpg'}, {'relative_path': 'two.jpg'}],
                    },
                    {
                        'group_id': 'near-a',
                        'duplicate_type': 'near',
                        'members': [{'relative_path': 'one.jpg'}, {'relative_path': 'three.jpg'}],
                    },
                    {
                        'group_id': 'visual-a',
                        'duplicate_type': 'visual',
                        'members': [{'relative_path': 'one.jpg'}, {'relative_path': 'two.jpg'}],
                    },
                    {
                        'group_id': 'shared',
                        'duplicate_type': 'exact',
                        'members': [{'relative_path': 'one.jpg'}, {'relative_path': 'two.jpg'}],
                    },
                ],
            },
        },
        'library-b': {'library_root': str(second_root), 'records': second_records},
        'invalid-library': 'invalid',
    })

    stats = client.get('/dashboard/stats').json()

    assert stats['total_images'] == 7
    assert stats['invalid_records'] == 1
    assert stats['invalid_libraries'] == 1
    assert stats['library_count'] == 2
    assert stats['processed_images'] == 2
    assert stats['processing'] == {
        'completed': 2, 'pending': 3, 'processing': 1, 'failed': 1, 'other': 0,
    }
    assert stats['classification'] == {
        'classified': 4, 'needs_review': 1, 'processing': 0, 'pending': 1,
        'failed': 1, 'other': 0,
    }
    assert stats['file_status'] == {
        'available': 3, 'missing': 1, 'modified': 1, 'unverified': 2,
    }
    categories = {item['category']: item for item in stats['categories']['distribution']}
    assert stats['categories']['total_classified'] == 4
    assert stats['categories']['categorized_images'] == 4
    assert stats['categories']['classified_without_category'] == 0
    assert categories['Government & Identity'] == {
        'category': 'Government & Identity', 'count': 2, 'percentage': 50.0,
    }
    assert categories['Education']['count'] == 2
    assert stats['duplicates'] == {
        'groups': 3,
        'exact_groups': 1,
        'near_duplicate_groups': 1,
        'visual_duplicate_groups': 1,
        'records_in_groups': 3,
    }
    libraries = {item['library_id']: item for item in stats['libraries']}
    assert libraries['library-a'] == {
        'library_id': 'library-a',
        'display_name': 'family_photos',
        'total_images': 5,
        'processed_images': 1,
        'missing_files': 1,
    }
    assert libraries['library-b']['total_images'] == 2
    assert 'library_root' not in libraries['library-a']


def test_dashboard_handles_empty_and_invalid_record_collections(dashboard_index):
    write_index(dashboard_index, {
        'empty': {'records': {}, 'duplicate_detection': {'groups': []}},
        'invalid-records': {'records': ['not', 'a', 'mapping']},
    })

    stats = client.get('/dashboard/stats').json()

    assert stats['library_count'] == 2
    assert stats['total_images'] == 0
    assert stats['invalid_records'] == 1


def test_folder_filter_returns_only_requested_library(dashboard_index, tmp_path):
    requested = tmp_path / 'requested'
    other = tmp_path / 'other'
    requested.mkdir()
    other.mkdir()
    write_index(dashboard_index, {
        'requested': {
            'library_root': str(requested),
            'records': {'one.jpg': {'file_status': 'available'}},
        },
        'other': {
            'library_root': str(other),
            'records': {'two.jpg': {'file_status': 'available'}},
        },
    })

    response = client.get('/dashboard/stats', params={'folder_path': str(requested)})

    assert response.status_code == 200
    assert response.json()['total_images'] == 1
    assert response.json()['library_count'] == 1
    assert response.json()['libraries'][0]['library_id'] == 'requested'


def test_malformed_index_is_reported_as_server_error(dashboard_index):
    dashboard_index.parent.mkdir(parents=True)
    dashboard_index.write_text('{invalid json', encoding='utf-8')

    response = client.get('/dashboard/stats')

    assert response.status_code == 500
    assert 'malformed' in response.json()['detail']


def test_dashboard_does_not_read_images_run_ai_or_change_phase_data(
    dashboard_index, tmp_path, monkeypatch,
):
    library_root = tmp_path / 'photos'
    library_root.mkdir()
    image_path = library_root / 'indexed.jpg'
    image_path.write_bytes(b'original image bytes')
    record = {
        'relative_path': 'indexed.jpg',
        'file_status': 'available',
        'ai_processing_status': 'completed',
        'ai_processing_version': 'phase2-local-v1',
        'ocr_text': 'existing OCR',
        'caption': 'existing caption',
        'embedding_path': 'existing.npy',
        'embedding_status': 'completed',
        'classification_status': 'classified',
        'category': 'Education',
        'classification_confidence': 0.82,
        'classification_version': 'phase3-multi-evidence-v1',
    }
    stored_index = {
        'version': 1,
        'libraries': {
            'library': {
                'library_root': str(library_root),
                'records': {'indexed.jpg': record},
            },
        },
    }
    write_index(dashboard_index, stored_index['libraries'])
    original_path_open = Path.open
    original_index = dashboard_index.read_bytes()
    original_image = image_path.read_bytes()

    monkeypatch.setattr(
        Path,
        'open',
        lambda self, *args, **kwargs: (
            pytest.fail('Dashboard attempted to open an original image')
            if self == image_path else original_path_open(self, *args, **kwargs)
        ),
    )
    monkeypatch.setattr(ai_processing_service, 'run_ocr', lambda *_: pytest.fail('OCR invoked'))
    monkeypatch.setattr(ai_processing_service, 'generate_caption', lambda *_: pytest.fail('BLIP invoked'))
    monkeypatch.setattr(ai_processing_service, 'generate_embedding', lambda *_args, **_kwargs: pytest.fail('OpenCLIP invoked'))
    monkeypatch.setattr(duplicate_service, 'process_library_duplicates', lambda *_args, **_kwargs: pytest.fail('Duplicate detection invoked'))

    response = client.get('/dashboard/stats')

    assert response.status_code == 200
    assert original_path_open(dashboard_index, 'rb').read() == original_index
    assert original_path_open(image_path, 'rb').read() == original_image
