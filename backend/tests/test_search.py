import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import library_service, search_service
from app.services.search_service import SearchConfig, search_images

client = TestClient(app)


@pytest.fixture
def search_index(tmp_path, monkeypatch):
    index_path = tmp_path / 'data' / 'index' / 'library_index.json'
    monkeypatch.setattr(search_service, 'INDEX_PATH', index_path)
    monkeypatch.setattr(library_service, 'INDEX_PATH', index_path)
    monkeypatch.setattr(search_service, 'encode_text_embeddings', lambda texts: np.tile(
        np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (len(texts), 1)
    ))
    monkeypatch.setattr(
        search_service,
        'rerank_candidates',
        lambda _query, candidates: {
            'status': 'disabled',
            'model': None,
            'order': [],
        },
    )
    return index_path


def write_record(library: Path, relative_path: str, embedding, **fields) -> dict:
    image_path = library / relative_path
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b'original image placeholder')
    embedding_path = library.parent / 'embeddings' / f'{Path(relative_path).stem}.npy'
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    if embedding is not None:
        np.save(embedding_path, embedding)
    return {
        'record_id': relative_path,
        'filename': Path(relative_path).name,
        'relative_path': relative_path,
        'extension': Path(relative_path).suffix,
        'width': 640,
        'height': 480,
        'file_status': 'available',
        'processing_status': 'pending',
        'ai_processing_status': 'completed',
        'embedding_status': 'completed' if embedding is not None else 'failed',
        'embedding_path': str(embedding_path) if embedding is not None else '',
        'classification_status': 'classified',
        'classification_confidence': 0.75,
        **fields,
    }


def save_index(path: Path, library: Path, records: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    library_id = library_service._library_id_for_path(str(library.resolve()))
    path.write_text(json.dumps({
        'version': 1,
        'libraries': {
            library_id: {
                'library_root': str(library.resolve()),
                'records': records,
            }
        },
    }), encoding='utf-8')


def fixture_records(tmp_path):
    library = tmp_path / 'library'
    library.mkdir()
    records = {
        'passport.jpg': write_record(
            library, 'passport.jpg', [1.0, 0.0, 0.0],
            ocr_text='Republic passport identity',
            caption='A passport document',
            category='Government & Identity',
            classification_confidence=0.88,
        ),
        'college.jpg': write_record(
            library, 'college.jpg', [0.8, 0.6, 0.0],
            ocr_text='University college certificate',
            caption='A college certificate',
            category='Education',
            classification_confidence=0.81,
        ),
        'meal.jpg': write_record(
            library, 'meal.jpg', [0.0, 1.0, 0.0],
            ocr_text='Menu dinner',
            caption='Food served at a restaurant',
            category='Food & Drinks',
            classification_confidence=0.77,
        ),
    }
    return library, records


def run_fixture_search(tmp_path, search_index, query, **kwargs):
    library, records = fixture_records(tmp_path)
    save_index(search_index, library, records)
    return search_images(query, folder_path=str(library), **kwargs)


def test_exact_filename_search(tmp_path, search_index):
    result = run_fixture_search(tmp_path, search_index, 'passport')

    assert result['total_results'] == 2
    assert result['results'][0]['filename'] == 'passport.jpg'
    assert 'filename' in result['results'][0]['matched_fields']


def test_ocr_keyword_search(tmp_path, search_index):
    result = run_fixture_search(tmp_path, search_index, 'identity')

    passport = next(item for item in result['results'] if item['filename'] == 'passport.jpg')
    assert 'ocr' in passport['matched_fields']
    assert 'identity' in passport['matched_keywords']


def test_caption_search(tmp_path, search_index):
    result = run_fixture_search(tmp_path, search_index, 'restaurant')

    meal = next(item for item in result['results'] if item['filename'] == 'meal.jpg')
    assert 'caption' in meal['matched_fields']


def test_related_sports_term_finds_cricket_filename(tmp_path, search_index):
    library, records = fixture_records(tmp_path)
    records['cricket.jpg'] = write_record(
        library,
        'cricket.jpg',
        [0.95, 0.05, 0.0],
        caption='A cricket player holding a bat and ball',
    )
    save_index(search_index, library, records)

    result = search_images('bat', folder_path=str(library))

    cricket = next(item for item in result['results'] if item['filename'] == 'cricket.jpg')
    assert 'filename' in cricket['matched_fields']
    assert 'cricket' in cricket['matched_keywords']


def test_related_ball_term_does_not_hide_caption_evidence(tmp_path, search_index):
    library, records = fixture_records(tmp_path)
    records['cricket.jpg'] = write_record(
        library,
        'cricket.jpg',
        [0.95, 0.05, 0.0],
        caption='A cricket player holding a ball',
    )
    save_index(search_index, library, records)

    result = search_images('ball', folder_path=str(library))

    cricket = next(item for item in result['results'] if item['filename'] == 'cricket.jpg')
    assert 'caption' in cricket['matched_fields']
    assert 'ball' in cricket['matched_keywords']


def test_category_search(tmp_path, search_index):
    result = run_fixture_search(tmp_path, search_index, 'education')

    college = next(item for item in result['results'] if item['filename'] == 'college.jpg')
    assert 'category' in college['matched_fields']


def test_semantic_search_uses_query_embedding_and_image_vectors(tmp_path, search_index, monkeypatch):
    calls = []

    def encode(texts):
        calls.append(texts)
        return np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(search_service, 'encode_text_embeddings', encode)
    result = run_fixture_search(tmp_path, search_index, 'a government identity document')

    assert calls == [['a government identity document']]
    assert result['results'][0]['filename'] == 'passport.jpg'
    assert result['results'][0]['semantic_score'] == pytest.approx(1.0)


def test_hybrid_ranking_combines_semantic_and_lexical_evidence(tmp_path, search_index):
    result = run_fixture_search(tmp_path, search_index, 'college certificate')

    assert result['results'][0]['filename'] == 'college.jpg'
    assert result['results'][0]['hybrid_score'] > result['results'][1]['hybrid_score']
    assert result['results'][0]['lexical_score'] > 0


def test_search_includes_low_confidence_review_candidates(tmp_path, search_index):
    library, records = fixture_records(tmp_path)
    records['uncertain-meal.jpg'] = write_record(
        library,
        'uncertain-meal.jpg',
        [1.0, 0.0, 0.0],
        caption='A plate of spicy noodles',
        category=None,
        top_category='Food & Drinks',
        classification_status='needs_review',
        classification_confidence=0.12,
    )
    save_index(search_index, library, records)

    result = search_images('spicy noodles', folder_path=str(library))

    meal = next(item for item in result['results'] if item['filename'] == 'uncertain-meal.jpg')
    assert meal['classification_status'] == 'needs_review'
    assert meal['category'] is None


def test_lexical_search_includes_records_without_embeddings(tmp_path, search_index):
    library, records = fixture_records(tmp_path)
    records['text-only.jpg'] = write_record(
        library,
        'text-only.jpg',
        None,
        caption='A red bicycle beside the garden gate',
    )
    save_index(search_index, library, records)

    result = search_images('bicycle', folder_path=str(library))

    bicycle = next(item for item in result['results'] if item['filename'] == 'text-only.jpg')
    assert bicycle['semantic_score'] is None
    assert 'caption' in bicycle['matched_fields']


def test_llm_reranking_controls_candidate_order(tmp_path, search_index, monkeypatch):
    library, records = fixture_records(tmp_path)
    save_index(search_index, library, records)
    monkeypatch.setattr(
        search_service,
        'rerank_candidates',
        lambda _query, candidates: {
            'status': 'used',
            'model': 'test-local-model',
            'order': ['1', '0'],
        },
    )

    result = search_images('passport document', folder_path=str(library))

    assert result['reranker']['status'] == 'used'
    assert result['results'][0]['filename'] == 'college.jpg'
    assert result['results'][0]['reranked_by_llm'] is True


def test_empty_query_is_rejected(search_index):
    with pytest.raises(ValueError, match='Search query is required'):
        search_images('  ')


def test_no_result_query_returns_empty_results(tmp_path, search_index, monkeypatch):
    library, records = fixture_records(tmp_path)
    save_index(search_index, library, records)
    monkeypatch.setattr(
        search_service,
        'encode_text_embeddings',
        lambda _texts: np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
    )
    result = search_images('nonexistentterm', folder_path=str(library))

    assert result['total_results'] == 0
    assert result['results'] == []


def test_weak_semantic_only_matches_are_filtered(tmp_path, search_index, monkeypatch):
    library, records = fixture_records(tmp_path)
    save_index(search_index, library, records)
    monkeypatch.setattr(
        search_service,
        'encode_text_embeddings',
        lambda _texts: np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
    )

    result = search_images('unrelated visual query', folder_path=str(library))

    assert result['results'] == []


def test_missing_embedding_is_skipped(tmp_path, search_index):
    library, records = fixture_records(tmp_path)
    records['without_embedding.jpg'] = write_record(
        library, 'without_embedding.jpg', None, filename='without_embedding.jpg'
    )
    save_index(search_index, library, records)

    result = search_images('passport', folder_path=str(library))

    assert all(item['filename'] != 'without_embedding.jpg' for item in result['results'])


def test_invalid_embedding_is_skipped_without_failing_search(tmp_path, search_index):
    library, records = fixture_records(tmp_path)
    bad_path = library.parent / 'embeddings' / 'invalid.npy'
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_bytes(b'invalid numpy data')
    records['invalid.jpg'] = write_record(
        library, 'invalid.jpg', None, filename='invalid.jpg',
        embedding_status='completed', embedding_path=str(bad_path),
    )
    save_index(search_index, library, records)

    result = search_images('passport', folder_path=str(library))

    assert result['total_results'] == 2


def test_embedding_vectors_are_cached_and_reloaded_when_file_changes(tmp_path, monkeypatch):
    path = tmp_path / 'cached.npy'
    np.save(path, np.array([1.0, 0.0], dtype=np.float32))
    search_service._load_embedding.cache_clear()
    original_load = np.load
    calls = []

    def tracked_load(*args, **kwargs):
        calls.append(args[0])
        return original_load(*args, **kwargs)

    monkeypatch.setattr(search_service.np, 'load', tracked_load)
    record = {'embedding_status': 'completed', 'embedding_path': str(path)}

    assert np.allclose(search_service._record_embedding(record), [1.0, 0.0])
    assert np.allclose(search_service._record_embedding(record), [1.0, 0.0])
    assert len(calls) == 1

    np.save(path, np.array([0.0, 1.0], dtype=np.float32))
    assert np.allclose(search_service._record_embedding(record), [0.0, 1.0])
    assert len(calls) == 2


def test_missing_file_is_skipped(tmp_path, search_index):
    library, records = fixture_records(tmp_path)
    (library / 'passport.jpg').unlink()
    save_index(search_index, library, records)

    result = search_images('passport', folder_path=str(library))

    assert all(item['filename'] != 'passport.jpg' for item in result['results'])


@pytest.mark.parametrize('file_status', ['modified', 'unverified'])
def test_stale_lifecycle_records_are_excluded_from_search(
    tmp_path, search_index, file_status,
):
    library, records = fixture_records(tmp_path)
    records['passport.jpg']['file_status'] = file_status
    save_index(search_index, library, records)

    result = search_images('passport', folder_path=str(library))

    assert all(item['filename'] != 'passport.jpg' for item in result['results'])


def test_search_limit(tmp_path, search_index, monkeypatch):
    monkeypatch.setattr(
        search_service,
        'encode_text_embeddings',
        lambda _texts: np.array([[1.0, 1.0, 0.0]], dtype=np.float32),
    )
    result = run_fixture_search(tmp_path, search_index, 'image', limit=2)

    assert result['returned_results'] == 2
    assert result['total_results'] == 3


def test_category_filter(tmp_path, search_index):
    result = run_fixture_search(
        tmp_path,
        search_index,
        'document',
        category='Education',
    )

    assert result['total_results'] == 1
    assert result['results'][0]['category'] == 'Education'


def test_multiple_libraries_can_be_searched_without_folder_filter(tmp_path, search_index, monkeypatch):
    library_one = tmp_path / 'library_one'
    library_two = tmp_path / 'library_two'
    library_one.mkdir()
    library_two.mkdir()
    first = write_record(library_one, 'first.jpg', [1.0, 0.0, 0.0], caption='first image')
    second = write_record(library_two, 'second.jpg', [0.0, 1.0, 0.0], caption='second image')
    search_index.parent.mkdir(parents=True, exist_ok=True)
    search_index.write_text(json.dumps({
        'version': 1,
        'libraries': {
            library_service._library_id_for_path(str(library_one.resolve())): {
                'library_root': str(library_one.resolve()), 'records': {'first.jpg': first},
            },
            library_service._library_id_for_path(str(library_two.resolve())): {
                'library_root': str(library_two.resolve()), 'records': {'second.jpg': second},
            },
        },
    }), encoding='utf-8')
    monkeypatch.setattr(
        search_service,
        'encode_text_embeddings',
        lambda _texts: np.array([[1.0, 1.0, 0.0]], dtype=np.float32),
    )

    result = search_images('image')

    assert result['total_results'] == 2


def test_search_reuses_existing_embeddings_and_never_runs_ai_image_services(tmp_path, search_index, monkeypatch):
    library, records = fixture_records(tmp_path)
    save_index(search_index, library, records)
    monkeypatch.setattr('app.services.ai_processing_service.run_ocr', lambda *_: pytest.fail('OCR was rerun'))
    monkeypatch.setattr('app.services.ai_processing_service.generate_caption', lambda *_: pytest.fail('BLIP was rerun'))
    monkeypatch.setattr('app.services.embedding_service.generate_embedding', lambda *_args, **_kwargs: pytest.fail('image embedding was regenerated'))

    result = search_images('passport', folder_path=str(library))

    assert result['total_results'] == 2


def test_search_api_returns_ranked_result_shape(tmp_path, search_index):
    library, records = fixture_records(tmp_path)
    save_index(search_index, library, records)

    response = client.post('/search', json={
        'folder_path': str(library),
        'query': 'passport',
        'limit': 1,
    })

    assert response.status_code == 200
    payload = response.json()
    assert payload['returned_results'] == 1
    result = payload['results'][0]
    assert {
        'record_id', 'filename', 'relative_path', 'category',
        'classification_confidence', 'semantic_score', 'lexical_score',
        'hybrid_score', 'matched_fields', 'matched_keywords',
        'search_reason', 'file_status', 'processing_status',
    } <= result.keys()


def test_search_api_rejects_empty_query():
    response = client.post('/search', json={'query': ' '})

    assert response.status_code == 400


def test_search_config_has_requested_default_weights():
    config = SearchConfig()

    assert config.weights() == {
        'semantic_score': 0.50,
        'ocr_score': 0.20,
        'filename_score': 0.10,
        'caption_score': 0.10,
        'category_score': 0.05,
        'metadata_score': 0.05,
    }
