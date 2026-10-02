import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.services import ai_processing_service, library_service
from app.services.library_service import scan_library

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolate_ai_state(tmp_path, monkeypatch):
    index_path = tmp_path / 'data' / 'index' / 'library_index.json'
    monkeypatch.setattr(library_service, 'INDEX_PATH', index_path)
    monkeypatch.setattr(library_service, 'INDEX_DIR', index_path.parent)
    monkeypatch.setattr(ai_processing_service, 'INDEX_PATH', index_path)
    monkeypatch.setattr(ai_processing_service, '_project_root', lambda: tmp_path)


def create_image(path: Path, size=(50, 40), color=(12, 34, 56)):
    image = Image.new('RGB', size=size, color=color)
    image.save(path)
    return path


def test_ai_services_import_cleanly():
    from app.services.caption_service import generate_caption
    from app.services.embedding_service import compute_embedding_path, generate_embedding
    from app.services.ocr_service import run_ocr

    assert callable(generate_caption)
    assert callable(run_ocr)
    assert callable(generate_embedding)
    assert callable(compute_embedding_path)


def test_ai_processing_rejects_missing_library_without_writing_index(tmp_path):
    missing_library = tmp_path / 'missing-library'

    with pytest.raises(FileNotFoundError, match='Folder does not exist'):
        ai_processing_service.process_library_images(str(missing_library))

    assert not library_service.INDEX_PATH.exists()


def test_ai_processing_stores_metadata_and_embedding_separately(tmp_path, monkeypatch):
    library_dir = tmp_path / 'library'
    library_dir.mkdir()
    image_path = create_image(library_dir / 'ai.jpg')
    scan_library(str(library_dir))

    monkeypatch.setattr(ai_processing_service, 'run_ocr', lambda img_path: {'text': 'hello world', 'confidence': 0.91, 'status': 'completed', 'error': None})
    monkeypatch.setattr(ai_processing_service, 'generate_caption', lambda img_path: {'caption': 'A test image', 'status': 'completed', 'error': None})

    def fake_generate_embedding(img_path, record_id, project_root=None):
        assert project_root == str(tmp_path)
        embedding_path = tmp_path / 'data' / 'embeddings' / f'{record_id}.npy'
        embedding_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(embedding_path, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        return {'embedding_path': str(embedding_path), 'embedding_status': 'completed', 'error': None}

    monkeypatch.setattr(ai_processing_service, 'generate_embedding', fake_generate_embedding)

    result = ai_processing_service.process_library_images(str(library_dir), force=True)

    assert result['processed'] == 1
    assert result['image_statuses'] == [{
        'record_id': 'ai.jpg',
        'relative_path': 'ai.jpg',
        'ai_processing_status': 'completed',
        'ai_processing_profile': 'detailed',
        'ocr_status': 'completed',
        'caption_status': 'completed',
        'embedding_status': 'completed',
    }]
    record = next(iter(next(iter(json.loads(
        library_service.INDEX_PATH.read_text(encoding='utf-8')
    )['libraries'].values()))['records'].values()))
    assert record['ocr_text'] == 'hello world'
    assert record['caption'] == 'A test image'
    assert record['ai_processing_status'] == 'completed'
    assert record['embedding_path'].endswith('.npy')
    assert 'data/embeddings' in record['embedding_path'].replace('\\', '/')
    assert str(image_path).replace('\\', '/') not in record['embedding_path']


def test_ai_reprocessing_is_skipped_for_unchanged_images(tmp_path, monkeypatch):
    library_dir = tmp_path / 'library'
    library_dir.mkdir()
    create_image(library_dir / 'skip.jpg')
    scan_library(str(library_dir))

    monkeypatch.setattr(ai_processing_service, 'run_ocr', lambda img_path: {'text': 'keep', 'confidence': 0.8, 'status': 'completed', 'error': None})
    monkeypatch.setattr(ai_processing_service, 'generate_caption', lambda img_path: {'caption': 'keep', 'status': 'completed', 'error': None})

    def fake_generate_embedding(img_path, record_id, project_root=None):
        assert project_root == str(tmp_path)
        embedding_path = tmp_path / 'data' / 'embeddings' / f'{record_id}.npy'
        embedding_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(embedding_path, np.array([0.1, 0.2, 0.3], dtype=np.float32))
        return {'embedding_path': str(embedding_path), 'embedding_status': 'completed', 'error': None}

    monkeypatch.setattr(ai_processing_service, 'generate_embedding', fake_generate_embedding)

    first = ai_processing_service.process_library_images(str(library_dir), force=True)
    second = ai_processing_service.process_library_images(str(library_dir))

    assert first['processed'] == 1
    assert second['skipped'] == 1
    assert second['processed'] == 0


def test_forced_ai_reprocessing_refreshes_without_erasing_existing_details(tmp_path, monkeypatch):
    library_dir = tmp_path / 'library'
    library_dir.mkdir()
    create_image(library_dir / 'refresh.jpg')
    scan_library(str(library_dir))

    monkeypatch.setattr(ai_processing_service, 'run_ocr', lambda _: {
        'text': 'original text', 'confidence': 0.9, 'status': 'completed', 'error': None,
    })
    monkeypatch.setattr(ai_processing_service, 'generate_caption', lambda _: {
        'caption': 'original caption', 'status': 'completed', 'error': None,
    })

    def fake_embedding(_, record_id, project_root=None):
        path = tmp_path / 'data' / 'embeddings' / f'{record_id}.npy'
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        return {'embedding_path': str(path), 'embedding_status': 'completed', 'error': None}

    monkeypatch.setattr(ai_processing_service, 'generate_embedding', fake_embedding)
    first = ai_processing_service.process_library_images(str(library_dir), force=True)
    assert first['processed'] == 1

    monkeypatch.setattr(ai_processing_service, 'run_ocr', lambda _: {
        'text': '', 'confidence': None, 'status': 'failed', 'error': 'OCR unavailable',
    })
    monkeypatch.setattr(ai_processing_service, 'generate_caption', lambda _: {
        'caption': '', 'status': 'failed', 'error': 'Caption unavailable',
    })
    refreshed = ai_processing_service.process_library_images(str(library_dir), force=True)

    assert refreshed['processed'] == 1
    record = next(iter(next(iter(json.loads(
        library_service.INDEX_PATH.read_text(encoding='utf-8')
    )['libraries'].values()))['records'].values()))
    assert record['ocr_text'] == 'original text'
    assert record['caption'] == 'original caption'
    assert record['embedding_status'] == 'completed'


def test_ai_failure_is_recorded_safely(tmp_path, monkeypatch):
    library_dir = tmp_path / 'library'
    library_dir.mkdir()
    create_image(library_dir / 'fail.jpg')
    scan_library(str(library_dir))

    monkeypatch.setattr(ai_processing_service, 'run_ocr', lambda img_path: {'text': '', 'confidence': None, 'status': 'failed', 'error': 'OCR unavailable'})
    monkeypatch.setattr(ai_processing_service, 'generate_caption', lambda img_path: {'caption': '', 'status': 'failed', 'error': 'Caption unavailable'})
    monkeypatch.setattr(ai_processing_service, 'generate_embedding', lambda img_path, record_id, project_root=None: {'embedding_path': '', 'embedding_status': 'failed', 'error': 'Embedding unavailable'})

    result = ai_processing_service.process_library_images(str(library_dir), force=True)

    assert result['failed'] == 1
    saved = json.loads(library_service.INDEX_PATH.read_text(encoding='utf-8'))
    record = next(iter(next(iter(saved['libraries'].values()))['records'].values()))
    assert record['ai_processing_status'] == 'failed'
    assert 'OCR unavailable' in (record['ai_processing_error'] or '')


def test_fast_profile_generates_embeddings_without_ocr_or_captions(tmp_path, monkeypatch):
    library_dir = tmp_path / 'library'
    library_dir.mkdir()
    create_image(library_dir / 'fast.jpg')
    scan_library(str(library_dir))

    def unexpected(*args, **kwargs):
        raise AssertionError('Fast processing must not run OCR or captioning.')

    monkeypatch.setattr(ai_processing_service, 'run_ocr', unexpected)
    monkeypatch.setattr(ai_processing_service, 'generate_caption', unexpected)

    def fake_generate_embedding(image_path, record_id, project_root=None):
        embedding_path = tmp_path / 'data' / 'embeddings' / f'{record_id}.npy'
        embedding_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(embedding_path, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        return {'embedding_path': str(embedding_path), 'embedding_status': 'completed', 'error': None}

    monkeypatch.setattr(ai_processing_service, 'generate_embedding', fake_generate_embedding)

    fast_result = ai_processing_service.process_library_images(str(library_dir), profile='fast')
    fast_record = fast_result['image_statuses'][0]
    assert fast_result['processed'] == 1
    assert fast_result['profile'] == 'fast'
    assert fast_record['ai_processing_profile'] == 'fast'
    assert fast_record['ocr_status'] == 'not_run'
    assert fast_record['caption_status'] == 'not_run'

    monkeypatch.setattr(ai_processing_service, 'run_ocr', lambda _: {
        'text': 'invoice total due', 'confidence': 0.95, 'status': 'completed', 'error': None,
    })
    monkeypatch.setattr(ai_processing_service, 'generate_caption', lambda _: {
        'caption': 'A paper receipt', 'status': 'completed', 'error': None,
    })
    detailed_result = ai_processing_service.process_library_images(str(library_dir), profile='detailed')

    assert detailed_result['processed'] == 1
    assert detailed_result['skipped'] == 0
    assert detailed_result['image_statuses'][0]['ai_processing_profile'] == 'detailed'


def test_caption_only_processing_generates_missing_caption_without_rerunning_other_models(
    tmp_path,
    monkeypatch,
):
    library_dir = tmp_path / 'library'
    library_dir.mkdir()
    create_image(library_dir / 'caption.jpg')
    scan_library(str(library_dir))

    def unexpected(*args, **kwargs):
        raise AssertionError('Caption-only processing must not run OCR or embeddings.')

    monkeypatch.setattr(ai_processing_service, 'run_ocr', unexpected)
    monkeypatch.setattr(ai_processing_service, 'generate_embedding', unexpected)
    caption_calls = []
    monkeypatch.setattr(ai_processing_service, 'generate_caption', lambda path: (
        caption_calls.append(path) or {
            'caption': 'A golden sunset over the ocean',
            'status': 'completed',
            'error': None,
        }
    ))

    result = ai_processing_service.process_library_captions(str(library_dir))
    repeated = ai_processing_service.process_library_captions(str(library_dir))

    assert result['processed'] == 1
    assert result['failed'] == 0
    assert result['image_statuses'][0]['caption'] == 'A golden sunset over the ocean'
    assert repeated['processed'] == 0
    assert repeated['skipped'] == 1
    assert len(caption_calls) == 1


def test_caption_only_failure_is_reported_and_can_be_retried(tmp_path, monkeypatch):
    library_dir = tmp_path / 'library'
    library_dir.mkdir()
    create_image(library_dir / 'caption.jpg')
    scan_library(str(library_dir))
    monkeypatch.setattr(ai_processing_service, 'generate_caption', lambda _: {
        'caption': '',
        'status': 'failed',
        'error': 'Caption model unavailable',
    })

    result = ai_processing_service.process_library_captions(str(library_dir))

    assert result['failed'] == 1
    assert result['processing_status'] == 'partial_failure'
    assert result['errors'] == [{
        'relative_path': 'caption.jpg',
        'error': 'Caption model unavailable',
    }]
    saved = json.loads(library_service.INDEX_PATH.read_text(encoding='utf-8'))
    record = next(iter(next(iter(saved['libraries'].values()))['records'].values()))
    assert record['caption_status'] == 'failed'
    assert record['caption_error'] == 'Caption model unavailable'
    assert not record.get('caption')


def test_caption_only_processing_reuses_caption_for_identical_indexed_image(tmp_path, monkeypatch):
    first_library = tmp_path / 'first-library'
    second_library = tmp_path / 'second-library'
    first_library.mkdir()
    second_library.mkdir()
    create_image(first_library / 'same.jpg')
    create_image(second_library / 'same.jpg')
    scan_library(str(first_library))
    scan_library(str(second_library))
    saved = json.loads(library_service.INDEX_PATH.read_text(encoding='utf-8'))
    first_id = library_service._library_id_for_path(str(first_library.resolve()))
    first_record = saved['libraries'][first_id]['records']['same.jpg']
    first_record.update({
        'caption': 'A dog playing in a garden',
        'caption_status': 'completed',
    })
    library_service.INDEX_PATH.write_text(json.dumps(saved), encoding='utf-8')
    monkeypatch.setattr(
        ai_processing_service,
        'generate_caption',
        lambda _: pytest.fail('Identical image should reuse its local caption.'),
    )

    result = ai_processing_service.process_library_captions(str(second_library))

    assert result['processed'] == 1
    assert result['image_statuses'][0]['caption'] == 'A dog playing in a garden'
    refreshed = json.loads(library_service.INDEX_PATH.read_text(encoding='utf-8'))
    second_id = library_service._library_id_for_path(str(second_library.resolve()))
    second_record = refreshed['libraries'][second_id]['records']['same.jpg']
    assert second_record['caption_status'] == 'completed'


def test_ai_api_passes_selected_profile_to_processor(monkeypatch):
    from app.routes import ai

    received = {}

    def fake_process_library_images(folder_path, force, record_ids, profile):
        received.update({
            'folder_path': folder_path,
            'force': force,
            'record_ids': record_ids,
            'profile': profile,
        })
        return {'profile': profile}

    monkeypatch.setattr(ai, 'process_library_images', fake_process_library_images)
    response = client.post('/ai/process', json={
        'folder_path': 'C:\\temporary-library',
        'record_ids': ['one.jpg'],
        'profile': 'fast',
    })

    assert response.status_code == 200
    assert response.json() == {'profile': 'fast'}
    assert received == {
        'folder_path': 'C:\\temporary-library',
        'force': False,
        'record_ids': ['one.jpg'],
        'profile': 'fast',
    }


def test_ai_captions_api_processes_only_requested_records(monkeypatch):
    from app.routes import ai

    received = {}

    def fake_process_library_captions(folder_path, record_ids, force):
        received.update({'folder_path': folder_path, 'record_ids': record_ids, 'force': force})
        return {'processed': 1}

    monkeypatch.setattr(ai, 'process_library_captions', fake_process_library_captions)
    response = client.post('/ai/captions', json={
        'folder_path': 'C:\\temporary-library',
        'record_ids': ['one.jpg'],
    })

    assert response.status_code == 200
    assert response.json() == {'processed': 1}
    assert received == {
        'folder_path': 'C:\\temporary-library',
        'record_ids': ['one.jpg'],
        'force': False,
    }
