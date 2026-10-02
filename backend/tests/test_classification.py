import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import classification_service, library_service
from app.services.classification_service import (
    CLASSIFICATION_VERSION,
    CATEGORY_PROMPTS,
    classify_record,
    process_library_classification,
)

client = TestClient(app)


@pytest.fixture
def classification_index(tmp_path, monkeypatch):
    index_path = tmp_path / 'data' / 'index' / 'library_index.json'
    monkeypatch.setattr(classification_service, 'INDEX_PATH', index_path)
    monkeypatch.setattr(library_service, 'INDEX_PATH', index_path)
    monkeypatch.setattr(library_service, 'INDEX_DIR', index_path.parent)
    monkeypatch.setattr(classification_service, '_semantic_category_scores', lambda record: {})
    return index_path


def make_record(filename='', ocr_text='', caption='', ocr_confidence=0.95):
    return {
        'filename': filename,
        'relative_path': filename,
        'ocr_text': ocr_text,
        'ocr_confidence': ocr_confidence,
        'ocr_status': 'completed',
        'caption': caption,
        'width': 800,
        'height': 600,
        'sha256': 'test-sha256',
    }


@pytest.mark.parametrize(
    ('record', 'expected'),
    [
        (make_record('aadhaar_passport.jpg', 'Government of India Aadhaar identity card'), 'Government & Identity'),
        (make_record('college_marksheet.png', 'University marksheet academic transcript'), 'Education'),
        (make_record('bank_transaction.png', 'Bank account transaction statement'), 'Finance'),
        (make_record('shop_invoice.jpg', 'Invoice purchase subtotal total due'), 'Bills & Receipts'),
        (make_record('flight_ticket.jpg', 'Airline flight boarding pass departure'), 'Travel'),
        (make_record('medical_prescription.jpg', 'Hospital prescription medicine patient'), 'Medical & Health'),
        (make_record('restaurant_menu.png', 'Restaurant menu food dinner'), 'Food & Drinks'),
        (make_record('phone_screenshot.png', 'Screenshot screen capture app screen'), 'Screenshots'),
    ],
)
def test_classifies_obvious_category_examples(record, expected):
    result = classify_record(record)

    assert result['classification_status'] == 'classified'
    assert result['category'] == expected
    assert result['classification_version'] == CLASSIFICATION_VERSION
    assert expected in result['category_scores']
    assert result['classification_reason']


def test_ambiguous_record_is_assigned_to_others():
    result = classify_record(make_record('image_001.jpg', '', '', ocr_confidence=None))

    assert result['classification_status'] == 'classified'
    assert result['category'] == 'Others'
    assert result['assignment_reason'] == 'Insufficient evidence; assigned to Others.'


def test_wide_landscape_dimensions_do_not_force_screenshot_category():
    record = make_record('photo_001.jpg', '', '', ocr_confidence=None)
    record.update({'width': 1920, 'height': 1080})

    result = classify_record(record)

    assert result['classification_status'] == 'classified'
    assert result['category'] == 'Others'
    assert result['top_category'] != 'Screenshots'
    assert 'metadata_score' not in result['evidence_sources']


def test_exactly_fifteen_required_categories_are_defined():
    assert list(CATEGORY_PROMPTS) == [
        'Government & Identity',
        'Education',
        'Medical & Health',
        'Finance',
        'Bills & Receipts',
        'Work & Professional',
        'Travel',
        'Events & Celebrations',
        'People & Family',
        'Nature & Places',
        'Animals & Pets',
        'Food & Drinks',
        'Screenshots',
        'Notes & Documents',
        'Others',
    ]


def test_conflicting_strong_evidence_is_assigned_to_others():
    record = make_record(
        'passport_aadhaar.jpg',
        'Invoice bill payment receipt amount due',
    )
    result = classify_record(record)

    assert result['classification_status'] == 'classified'
    assert result['category'] == 'Others'
    assert any('different categories' in reason for reason in result['classification_reason'])
    assert result['review_reasons'] == []
    assert result['assignment_reason'] == 'Evidence sources conflict; assigned to Others.'
    assert result['conflicting_sources']


def test_close_semantic_categories_need_review(monkeypatch, tmp_path):
    embedding_path = tmp_path / 'embedding.npy'
    embedding_path.write_bytes(b'placeholder')
    monkeypatch.setattr(
        classification_service,
        '_semantic_category_scores',
        lambda record: {
            'Government & Identity': 0.51,
            'Notes & Documents': 0.50,
            **{category: 0.01 for category in CATEGORY_PROMPTS
               if category not in {'Government & Identity', 'Notes & Documents'}},
        },
    )
    record = make_record('unclear.jpg', '', '', ocr_confidence=None)
    record.update({'embedding_status': 'completed', 'embedding_path': str(embedding_path)})

    result = classify_record(record, classification_service._semantic_category_scores(record))

    assert result['classification_status'] == 'classified'
    assert result['category'] == 'Others'
    assert result['review_reasons'] == []
    assert result['assignment_reason'] == 'Leading categories are too close; assigned to Others.'


def test_categories_with_clear_relative_lead_are_classified(monkeypatch, tmp_path):
    embedding_path = tmp_path / 'embedding.npy'
    embedding_path.write_bytes(b'placeholder')
    monkeypatch.setattr(
        classification_service,
        '_semantic_category_scores',
        lambda record: {
            'Food & Drinks': 0.30,
            'Events & Celebrations': 0.26,
            **{category: 0.01 for category in CATEGORY_PROMPTS
               if category not in {'Food & Drinks', 'Events & Celebrations'}},
        },
    )
    record = make_record('food.jpg', '', '', ocr_confidence=None)
    record.update({'embedding_status': 'completed', 'embedding_path': str(embedding_path)})

    result = classify_record(record, classification_service._semantic_category_scores(record))

    assert result['classification_status'] == 'classified'
    assert result['category'] == 'Food & Drinks'


def test_similarly_scored_categories_are_assigned_to_others(monkeypatch, tmp_path):
    embedding_path = tmp_path / 'embedding.npy'
    embedding_path.write_bytes(b'placeholder')
    monkeypatch.setattr(
        classification_service,
        '_semantic_category_scores',
        lambda record: {
            'Food & Drinks': 0.30,
            'Events & Celebrations': 0.28,
            **{category: 0.01 for category in CATEGORY_PROMPTS
               if category not in {'Food & Drinks', 'Events & Celebrations'}},
        },
    )
    record = make_record('unclear.jpg', '', '', ocr_confidence=None)
    record.update({'embedding_status': 'completed', 'embedding_path': str(embedding_path)})

    result = classify_record(record, classification_service._semantic_category_scores(record))

    assert result['classification_status'] == 'classified'
    assert result['category'] == 'Others'
    assert result['review_reasons'] == []


def test_clear_semantic_winner_is_classified(monkeypatch, tmp_path):
    embedding_path = tmp_path / 'embedding.npy'
    embedding_path.write_bytes(b'placeholder')
    monkeypatch.setattr(
        classification_service,
        '_semantic_category_scores',
        lambda record: {
            'Government & Identity': 0.75,
            'Notes & Documents': 0.10,
            **{category: 0.01 for category in CATEGORY_PROMPTS
               if category not in {'Government & Identity', 'Notes & Documents'}},
        },
    )
    record = make_record('unclear.jpg', '', '', ocr_confidence=None)
    record.update({'embedding_status': 'completed', 'embedding_path': str(embedding_path)})

    result = classify_record(record, classification_service._semantic_category_scores(record))

    assert result['classification_status'] == 'classified'
    assert result['category'] == 'Government & Identity'


def test_weak_category_evidence_is_assigned_to_others(monkeypatch, tmp_path):
    embedding_path = tmp_path / 'embedding.npy'
    embedding_path.write_bytes(b'placeholder')
    scores = {
        'Food & Drinks': 0.10,
        **{category: 0.01 for category in CATEGORY_PROMPTS if category != 'Food & Drinks'},
    }
    monkeypatch.setattr(classification_service, '_semantic_category_scores', lambda _record: scores)
    record = make_record('unclear.jpg', '', '', ocr_confidence=None)
    record.update({'embedding_status': 'completed', 'embedding_path': str(embedding_path)})

    result = classify_record(record, scores)

    assert result['classification_status'] == 'classified'
    assert result['category'] == 'Others'
    assert 'Weak evidence' in result['assignment_reason']


def test_specific_visual_concept_breaks_food_category_tie():
    record = make_record('images (35).jpg', '', '', ocr_confidence=None)
    semantic_scores = {
        'Food & Drinks': 0.1381,
        'Events & Celebrations': 0.1116,
        **{category: 0.01 for category in CATEGORY_PROMPTS
           if category not in {'Food & Drinks', 'Events & Celebrations'}},
    }
    semantic_concepts = {
        'Food & Drinks': [{'concept': 'food plate', 'score': 0.242}],
        'Events & Celebrations': [{'concept': 'party', 'score': 0.10}],
    }

    result = classify_record(record, semantic_scores, semantic_concepts)

    assert result['classification_status'] == 'classified'
    assert result['category'] == 'Food & Drinks'


@pytest.mark.parametrize(
    'caption',
    [
        'a squirrel is reaching its head out to the camera',
        'a baby monkey eating a piece of fruit',
        'a puppy with a bowl of food in its mouth',
    ],
)
def test_clear_animal_caption_is_classified_as_animals(caption):
    result = classify_record(make_record('image.jpg', '', caption, ocr_confidence=None))

    assert result['classification_status'] == 'classified'
    assert result['category'] == 'Animals & Pets'


def test_generic_filename_does_not_override_strong_caption_evidence():
    result = classify_record(make_record(
        'IMG_1234.jpg',
        '',
        'a mountain landscape with a lake and trees',
        ocr_confidence=None,
    ))

    assert result['matched_concepts']['Nature & Places']
    assert result['category'] == 'Nature & Places'


def test_classification_returns_secondary_categories_from_real_scores():
    result = classify_record(make_record(
        'wedding.jpg',
        '',
        'bride and groom at a wedding ceremony with family',
        ocr_confidence=None,
    ))

    assert result['category'] == 'Events & Celebrations'
    assert 'People & Family' in result['secondary_categories']
    assert result['top_categories'][0]['category'] == 'Events & Celebrations'


def test_category_text_embedding_scores_are_cached_and_use_phase2_image_embedding(tmp_path, monkeypatch):
    image_embedding_path = tmp_path / 'image.npy'
    import numpy as np

    np.save(image_embedding_path, np.array([1.0, 0.0], dtype=np.float32))
    category_vectors = np.zeros((len(CATEGORY_PROMPTS), 2), dtype=np.float32)
    category_vectors[0] = [1.0, 0.0]
    monkeypatch.setattr(classification_service, '_category_text_embeddings', category_vectors)
    result = classification_service._semantic_category_scores({
        'embedding_status': 'completed',
        'embedding_path': str(image_embedding_path),
    })

    assert len(result) == 15
    assert max(result.values()) > 0.99


def test_classification_api_updates_index_and_skips_valid_results(tmp_path, classification_index):
    library = tmp_path / 'library'
    library.mkdir()
    (library / 'invoice.jpg').write_bytes(b'not used by classifier')
    library_id = library_service._library_id_for_path(str(library.resolve()))
    classification_index.parent.mkdir(parents=True, exist_ok=True)
    classification_index.write_text(json.dumps({
        'version': 1,
        'libraries': {
            library_id: {
                'library_root': str(library.resolve()),
                'records': {
                    'invoice.jpg': {
                        'record_id': 'invoice.jpg',
                        'filename': 'invoice.jpg',
                        'relative_path': 'invoice.jpg',
                        'file_status': 'available',
                        'sha256': 'invoice-hash',
                        'ocr_status': 'completed',
                        'ocr_text': 'Invoice total due',
                        'ocr_confidence': 0.95,
                        'caption': '',
                        'width': 100,
                        'height': 100,
                    }
                },
            }
        },
    }), encoding='utf-8')

    response = client.post('/classification/process', json={'folder_path': str(library)})
    assert response.status_code == 200
    assert response.json()['classified'] == 1
    classification = response.json()['classifications'][0]
    assert classification['ocr_text'] == 'Invoice total due'
    assert classification['ocr_confidence'] == 0.95
    assert classification['ocr_status'] == 'completed'
    assert classification['evidence_sources']
    assert 'category_scores' in classification
    saved = json.loads(classification_index.read_text(encoding='utf-8'))
    record = saved['libraries'][library_id]['records']['invoice.jpg']
    assert record['category'] == 'Bills & Receipts'
    assert record['classification_status'] == 'classified'

    skipped = process_library_classification(str(library))
    assert skipped['skipped'] == 1
    assert skipped['classified'] == 1
    assert skipped['needs_review'] == 0
    skipped_classification = skipped['classifications'][0]
    assert skipped_classification['ocr_text'] == 'Invoice total due'
    assert 'category_scores' in skipped_classification

    saved['libraries'][library_id]['records']['invoice.jpg']['sha256'] = 'changed-hash'
    classification_index.write_text(json.dumps(saved), encoding='utf-8')
    reclassified = process_library_classification(str(library))
    assert reclassified['classified'] == 1
    assert reclassified['skipped'] == 0


def test_existing_review_result_is_re_evaluated(
    tmp_path,
    classification_index,
):
    library = tmp_path / 'library'
    library.mkdir()
    library_id = library_service._library_id_for_path(str(library.resolve()))
    classification_index.parent.mkdir(parents=True, exist_ok=True)
    classification_index.write_text(json.dumps({
        'version': 1,
        'libraries': {
            library_id: {
                'library_root': str(library.resolve()),
                'records': {
                    'invoice.jpg': {
                        'record_id': 'invoice.jpg',
                        'filename': 'invoice.jpg',
                        'relative_path': 'invoice.jpg',
                        'file_status': 'available',
                        'sha256': 'invoice-hash',
                        'ocr_status': 'completed',
                        'ocr_text': 'Invoice total due',
                        'ocr_confidence': 0.95,
                        'classification_status': 'needs_review',
                        'classification_version': CLASSIFICATION_VERSION,
                        'classification_sha256': 'invoice-hash',
                        'category_scores': {'Bills & Receipts': 0.2},
                        'classification_reason': ['Old low-confidence result'],
                    },
                },
            },
        },
    }), encoding='utf-8')

    result = process_library_classification(str(library))

    assert result['classified'] == 1
    assert result['needs_review'] == 0
    assert result['skipped'] == 0


def test_reviewed_image_can_be_assigned_to_any_category(tmp_path, classification_index):
    library = tmp_path / 'library'
    library.mkdir()
    library_id = library_service._library_id_for_path(str(library.resolve()))
    classification_index.parent.mkdir(parents=True, exist_ok=True)
    classification_index.write_text(json.dumps({
        'version': 1,
        'libraries': {
            library_id: {
                'library_root': str(library.resolve()),
                'records': {
                    'unclear.jpg': {
                        'record_id': 'unclear.jpg',
                        'filename': 'unclear.jpg',
                        'relative_path': 'unclear.jpg',
                        'file_status': 'available',
                        'sha256': 'unclear-hash',
                        'classification_status': 'needs_review',
                        'top_category': 'Others',
                    },
                },
            },
        },
    }), encoding='utf-8')

    result = classification_service.approve_classification(
        str(library),
        'unclear.jpg',
        'Animals & Pets',
    )

    assert result['classification_status'] == 'classified'
    assert result['category'] == 'Animals & Pets'


def test_classification_service_records_failure_without_invalidating_index(
    tmp_path,
    classification_index,
    monkeypatch,
):
    library = tmp_path / 'library'
    library.mkdir()
    library_id = library_service._library_id_for_path(str(library.resolve()))
    classification_index.parent.mkdir(parents=True, exist_ok=True)
    classification_index.write_text(json.dumps({
        'version': 1,
        'libraries': {
            library_id: {
                'library_root': str(library.resolve()),
                'records': {
                    'broken.jpg': {
                        'record_id': 'broken.jpg',
                        'filename': 'broken.jpg',
                        'relative_path': 'broken.jpg',
                        'file_status': 'available',
                        'sha256': 'broken-hash',
                        'embedding_status': 'completed',
                        'embedding_path': str(tmp_path / 'missing.npy'),
                    }
                },
            }
        },
    }), encoding='utf-8')

    def fail_semantics(_record):
        raise ValueError('controlled embedding read failure')

    monkeypatch.setattr(classification_service, '_semantic_category_scores', fail_semantics)
    result = process_library_classification(str(library))

    saved = json.loads(classification_index.read_text(encoding='utf-8'))
    record = saved['libraries'][library_id]['records']['broken.jpg']
    assert result['failed'] == 1
    assert record['classification_status'] == 'failed'
    assert record['classification_error'] == 'controlled embedding read failure'
    assert saved['version'] == 1
    assert record['sha256'] == 'broken-hash'
