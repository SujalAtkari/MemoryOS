import hashlib
import json
import time
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont
import pytest

from app.main import app
from app.services import (
    ai_processing_service,
    classification_service,
    dashboard_service,
    duplicate_service,
    file_operations_service,
    lifecycle_service,
    library_service,
    search_service,
)

client = TestClient(app)


def _measure_phase(timings, name, operation):
    started = time.perf_counter()
    result = operation()
    timings[name] = round(time.perf_counter() - started, 6)
    return result


def test_real_scan_ai_classification_and_semantic_search(tmp_path, monkeypatch):
    phase_timings = {}
    library = tmp_path / 'library'
    library.mkdir()
    image_path = library / 'passport_document.jpg'
    image = Image.new('RGB', (800, 300), 'white')
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 58)
    except OSError:
        font = ImageFont.load_default()
    draw.text((28, 45), 'GOVERNMENT OF INDIA', fill='black', font=font)
    draw.text((28, 125), 'PASSPORT IDENTITY DOCUMENT', fill='black', font=font)
    draw.rectangle((620, 40, 760, 250), fill=(35, 95, 160))
    image.save(image_path)
    original_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()

    index_path = tmp_path / 'data' / 'index' / 'library_index.json'
    monkeypatch.setattr(library_service, 'INDEX_PATH', index_path)
    monkeypatch.setattr(library_service, 'INDEX_DIR', index_path.parent)
    monkeypatch.setattr(ai_processing_service, 'INDEX_PATH', index_path)
    monkeypatch.setattr(ai_processing_service, '_project_root', lambda: tmp_path)
    monkeypatch.setattr(classification_service, 'INDEX_PATH', index_path)
    monkeypatch.setattr(dashboard_service, 'INDEX_PATH', index_path)
    monkeypatch.setattr(duplicate_service, 'INDEX_PATH', index_path)
    monkeypatch.setattr(file_operations_service, 'INDEX_PATH', index_path)
    monkeypatch.setattr(lifecycle_service, 'INDEX_PATH', index_path)
    monkeypatch.setattr(search_service, 'INDEX_PATH', index_path)

    scanned = _measure_phase(
        phase_timings,
        'phase1_scan',
        lambda: client.post('/library/scan', json={'folder_path': str(library)}),
    )
    assert scanned.status_code == 200
    processed = _measure_phase(
        phase_timings,
        'phase2_ai_processing_one_real_image',
        lambda: client.post('/ai/process', json={'folder_path': str(library)}),
    )
    assert processed.status_code == 200
    assert processed.json()['processed'] == 1
    classified = _measure_phase(
        phase_timings,
        'phase3_classification',
        lambda: client.post('/classification/process', json={'folder_path': str(library)}),
    )
    assert classified.status_code == 200

    index_before_search = json.loads(index_path.read_text(encoding='utf-8'))
    library_record = next(iter(next(iter(index_before_search['libraries'].values()))['records'].values()))
    assert library_record['ocr_status'] == 'completed'
    assert library_record['caption_status'] == 'completed'
    assert library_record['embedding_status'] == 'completed'
    assert library_record['classification_status'] == 'classified'
    stored_embedding_path = Path(library_record['embedding_path'])
    assert stored_embedding_path.is_file()
    if library_record['classification_status'] == 'classified':
        library_record['classification_confidence'] = 0.9
        library_record['confidence'] = 0.9
        index_path.write_text(json.dumps(index_before_search), encoding='utf-8')
    phase2_before = {
        key: library_record.get(key)
        for key in (
            'ocr_text', 'ocr_confidence', 'caption', 'embedding_path',
            'embedding_status', 'ai_processing_version', 'ai_processing_status',
        )
    }
    phase3_before = {
        key: library_record.get(key)
        for key in (
            'category', 'classification_confidence', 'classification_status',
            'classification_version', 'category_scores',
        )
    }
    phase1_before = {
        key: library_record.get(key)
        for key in (
            'record_id', 'relative_path', 'filename', 'extension', 'file_size',
            'modified_time', 'width', 'height', 'aspect_ratio', 'sha256',
            'phash', 'file_status',
        )
    }

    monkeypatch.setattr(ai_processing_service, 'run_ocr', lambda *_args: (_ for _ in ()).throw(AssertionError('Search reran OCR')))
    monkeypatch.setattr(ai_processing_service, 'generate_caption', lambda *_args: (_ for _ in ()).throw(AssertionError('Search reran BLIP')))
    monkeypatch.setattr(ai_processing_service, 'generate_embedding', lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError('Search regenerated image embedding')))

    search_response = _measure_phase(
        phase_timings,
        'phase4_search_one_ai_processed_record',
        lambda: client.post('/search', json={
            'folder_path': str(library),
            'query': 'government passport identity document',
            'limit': 5,
            'use_llm': False,
        }),
    )
    assert search_response.status_code == 200
    search_payload = search_response.json()
    assert search_payload['total_results'] == 1
    result = search_payload['results'][0]
    assert result['relative_path'] == 'passport_document.jpg'
    assert result['semantic_score'] > 0
    assert result['hybrid_score'] > 0

    repeated_search = client.post('/search', json={
        'folder_path': str(library),
        'query': 'government passport identity document',
        'limit': 5,
    })
    assert repeated_search.status_code == 200
    assert repeated_search.json()['results'][0]['record_id'] == result['record_id']

    embedding_bytes_before = stored_embedding_path.read_bytes()
    exact_copy_path = library / 'passport_document_copy.jpg'
    exact_copy_path.write_bytes(image_path.read_bytes())
    rescanned = client.post('/library/scan', json={'folder_path': str(library)})
    assert rescanned.status_code == 200
    duplicate_response = _measure_phase(
        phase_timings,
        'phase5_duplicate_detection_two_records',
        lambda: client.post('/duplicates/process', json={
            'folder_path': str(library),
            'include_visual': True,
        }),
    )
    assert duplicate_response.status_code == 200
    duplicate_payload = duplicate_response.json()
    assert duplicate_payload['exact_groups'] == 1
    assert duplicate_payload['processed'] == 2
    assert duplicate_payload['groups'][0]['duplicate_type'] == 'exact'
    repeated_duplicates = client.post('/duplicates/process', json={
        'folder_path': str(library),
        'include_visual': True,
    })
    assert repeated_duplicates.status_code == 200
    assert repeated_duplicates.json()['processed'] == 0
    assert repeated_duplicates.json()['skipped'] == 2

    index_before_dashboard = index_path.read_bytes()
    index_snapshot = json.loads(index_before_dashboard)
    indexed_record = index_snapshot['libraries'][scanned.json()['library_id']]['records']['passport_document.jpg']
    monkeypatch.setattr(
        duplicate_service,
        'process_library_duplicates',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError('Dashboard reran duplicate detection')
        ),
    )
    dashboard_response = _measure_phase(
        phase_timings,
        'phase6_dashboard_two_records',
        lambda: client.get('/dashboard/stats', params={'folder_path': str(library)}),
    )
    assert dashboard_response.status_code == 200
    dashboard = dashboard_response.json()
    assert dashboard['total_images'] == 2
    assert dashboard['processed_images'] == 1
    assert dashboard['processing']['completed'] == 1
    assert dashboard['processing']['pending'] == 1
    assert dashboard['classification'][indexed_record['classification_status']] == 1
    assert dashboard['classification']['pending'] == 1
    assert dashboard['file_status']['available'] == 2
    assert dashboard['duplicates']['groups'] == duplicate_payload['exact_groups'] == 1
    assert dashboard['duplicates']['exact_groups'] == 1
    assert dashboard['duplicates']['records_in_groups'] == 2
    is_classified = indexed_record['classification_status'] == 'classified'
    has_category = indexed_record.get('category') in {
        item['category'] for item in dashboard['categories']['distribution']
    }
    assert dashboard['categories']['total_classified'] == int(is_classified)
    assert dashboard['categories']['categorized_images'] == int(is_classified and has_category)
    assert dashboard['libraries'][0]['total_images'] == 2
    assert dashboard['libraries'][0]['missing_files'] == 0
    assert index_path.read_bytes() == index_before_dashboard
    lifecycle_check = _measure_phase(
        phase_timings,
        'phase8_unchanged_check_two_records',
        lambda: client.post('/lifecycle/check', json={
            'library_id': scanned.json()['library_id'],
        }),
    )
    assert lifecycle_check.status_code == 200
    assert lifecycle_check.json()['available'] == 2
    assert lifecycle_check.json()['modified'] == 0
    assert lifecycle_check.json()['missing'] == 0
    assert index_path.read_bytes() == index_before_dashboard

    index_after_duplicates = json.loads(index_path.read_text(encoding='utf-8'))
    record_after_search = index_after_duplicates['libraries'][scanned.json()['library_id']]['records']['passport_document.jpg']
    assert {
        key: record_after_search.get(key) for key in phase2_before
    } == phase2_before
    assert {
        key: record_after_search.get(key) for key in phase3_before
    } == phase3_before
    assert {
        key: record_after_search.get(key) for key in phase1_before
    } == phase1_before
    assert stored_embedding_path.read_bytes() == embedding_bytes_before
    assert hashlib.sha256(image_path.read_bytes()).hexdigest() == original_sha256
    assert image_path.is_file()
    assert exact_copy_path.read_bytes() == image_path.read_bytes()
    assert not (tmp_path / 'data' / image_path.name).exists()

    renamed = _measure_phase(
        phase_timings,
        'phase7_rename',
        lambda: client.post('/files/rename', json={
            'library_id': scanned.json()['library_id'],
            'record_id': 'passport_document.jpg',
            'new_filename': 'renamed-passport',
        }),
    )
    assert renamed.status_code == 200
    renamed_path = library / 'renamed-passport.jpg'
    assert not image_path.exists()
    assert hashlib.sha256(renamed_path.read_bytes()).hexdigest() == original_sha256
    assert 'embedding_path' not in renamed.json()['record']
    renamed_record = json.loads(index_path.read_text(encoding='utf-8'))[
        'libraries'
    ][scanned.json()['library_id']]['records']['renamed-passport.jpg']
    assert renamed_record['embedding_path'] == str(stored_embedding_path)
    assert renamed_record['ocr_text'] == phase2_before['ocr_text']
    assert renamed_record['category'] == phase3_before['category']

    search_after_rename = client.post('/search', json={
        'folder_path': str(library),
        'query': 'government passport identity document',
        'limit': 5,
    })
    assert search_after_rename.status_code == 200
    assert any(
        result['relative_path'] == 'renamed-passport.jpg'
        for result in search_after_rename.json()['results']
    )

    archive = library / 'archive'
    archive.mkdir()
    moved = _measure_phase(
        phase_timings,
        'phase7_move',
        lambda: client.post('/files/move', json={
            'library_id': scanned.json()['library_id'],
            'record_id': 'renamed-passport.jpg',
            'destination_folder': 'archive',
        }),
    )
    assert moved.status_code == 200
    moved_path = archive / 'renamed-passport.jpg'
    assert not renamed_path.exists()
    assert hashlib.sha256(moved_path.read_bytes()).hexdigest() == original_sha256
    moved_record = json.loads(index_path.read_text(encoding='utf-8'))[
        'libraries'
    ][scanned.json()['library_id']]['records']['archive/renamed-passport.jpg']
    assert moved_record['embedding_path'] == str(stored_embedding_path)

    search_after_move = client.post('/search', json={
        'folder_path': str(library),
        'query': 'government passport identity document',
        'limit': 5,
    })
    assert search_after_move.status_code == 200
    assert any(
        result['relative_path'] == 'archive/renamed-passport.jpg'
        for result in search_after_move.json()['results']
    )

    def unexpected_processing(*_args, **_kwargs):
        pytest.fail('Lifecycle checks must not rerun AI, classification, or duplicate processing')

    monkeypatch.setattr(ai_processing_service, 'process_library_images', unexpected_processing)
    monkeypatch.setattr(ai_processing_service, 'run_ocr', unexpected_processing)
    monkeypatch.setattr(ai_processing_service, 'generate_caption', unexpected_processing)
    monkeypatch.setattr(ai_processing_service, 'generate_embedding', unexpected_processing)
    monkeypatch.setattr(classification_service, 'process_library_classification', unexpected_processing)
    monkeypatch.setattr(duplicate_service, 'process_library_duplicates', unexpected_processing)

    original_copy_bytes = exact_copy_path.read_bytes()
    Image.new('RGB', (800, 300), (210, 30, 70)).save(exact_copy_path, format='JPEG')
    modified_check = _measure_phase(
        phase_timings,
        'phase8_modified_check_two_records',
        lambda: client.post('/lifecycle/check', json={
            'library_id': scanned.json()['library_id'],
        }),
    )
    assert modified_check.status_code == 200
    assert modified_check.json()['modified'] == 1
    assert modified_check.json()['available'] == 1
    assert modified_check.json()['missing'] == 0
    modified_index = json.loads(index_path.read_text(encoding='utf-8'))
    assert (
        modified_index['libraries'][scanned.json()['library_id']]
        ['records']['passport_document_copy.jpg']['file_status']
        == 'modified'
    )

    moved_path.unlink()
    relocation_path = library / 'recovered' / 'passport_document.jpg'
    relocation_path.parent.mkdir()
    relocation_path.write_bytes(original_copy_bytes)
    missing_check = _measure_phase(
        phase_timings,
        'phase8_missing_and_relocation_check_two_records',
        lambda: client.post('/lifecycle/check', json={
            'library_id': scanned.json()['library_id'],
        }),
    )
    assert missing_check.status_code == 200
    lifecycle_result = missing_check.json()
    assert lifecycle_result['modified'] == 1
    assert lifecycle_result['missing'] == 1
    assert lifecycle_result['available'] == 0
    assert lifecycle_result['relocations_detected'] == 1
    relocation = next(
        result for result in lifecycle_result['possible_relocations']
        if result['record_id'] == 'archive/renamed-passport.jpg'
    )
    assert relocation['possible_relocations'] == ['recovered/passport_document.jpg']

    before_reconcile = json.loads(index_path.read_text(encoding='utf-8'))
    indexed_library = before_reconcile['libraries'][scanned.json()['library_id']]
    assert 'archive/renamed-passport.jpg' in indexed_library['records']
    assert indexed_library['records']['archive/renamed-passport.jpg']['file_status'] == 'missing'
    assert 'recovered/passport_document.jpg' not in indexed_library['records']
    assert indexed_library['records']['archive/renamed-passport.jpg']['sha256'] == original_sha256
    for key, value in phase2_before.items():
        assert indexed_library['records']['archive/renamed-passport.jpg'].get(key) == value
    for key, value in phase3_before.items():
        assert indexed_library['records']['archive/renamed-passport.jpg'].get(key) == value
    assert stored_embedding_path.is_file()
    assert stored_embedding_path.read_bytes() == embedding_bytes_before

    dashboard_before_reconcile = client.get(
        '/dashboard/stats',
        params={'folder_path': str(library)},
    )
    assert dashboard_before_reconcile.status_code == 200
    lifecycle_counts = dashboard_before_reconcile.json()['file_status']
    assert lifecycle_counts['available'] == 0
    assert lifecycle_counts['modified'] == 1
    assert lifecycle_counts['missing'] == 1

    reconciled = _measure_phase(
        phase_timings,
        'phase8_reconcile',
        lambda: client.post('/lifecycle/reconcile', json={
            'library_id': scanned.json()['library_id'],
            'record_id': 'archive/renamed-passport.jpg',
            'new_relative_path': 'recovered/passport_document.jpg',
            'confirm': True,
        }),
    )
    assert reconciled.status_code == 200
    after_reconcile = json.loads(index_path.read_text(encoding='utf-8'))
    reconciled_library = after_reconcile['libraries'][scanned.json()['library_id']]
    assert 'archive/renamed-passport.jpg' not in reconciled_library['records']
    reconciled_record = reconciled_library['records']['recovered/passport_document.jpg']
    assert reconciled_record['relative_path'] == 'recovered/passport_document.jpg'
    assert reconciled_record['sha256'] == original_sha256
    for key, value in phase2_before.items():
        assert reconciled_record.get(key) == value
    for key, value in phase3_before.items():
        assert reconciled_record.get(key) == value
    assert hashlib.sha256(relocation_path.read_bytes()).hexdigest() == original_sha256
    assert stored_embedding_path.read_bytes() == embedding_bytes_before

    dashboard_after_reconcile = client.get(
        '/dashboard/stats',
        params={'folder_path': str(library)},
    )
    assert dashboard_after_reconcile.status_code == 200
    reconciled_counts = dashboard_after_reconcile.json()['file_status']
    assert reconciled_counts['available'] == 1
    assert reconciled_counts['modified'] == 1
    assert reconciled_counts['missing'] == 0

    deleted = _measure_phase(
        phase_timings,
        'phase7_confirmed_delete',
        lambda: client.post('/files/delete', json={
            'library_id': scanned.json()['library_id'],
            'record_id': 'recovered/passport_document.jpg',
            'confirm': True,
        }),
    )
    assert deleted.status_code == 200
    assert not moved_path.exists()
    assert not relocation_path.exists()
    final_index = json.loads(index_path.read_text(encoding='utf-8'))
    final_library = final_index['libraries'][scanned.json()['library_id']]
    assert final_library['records']['recovered/passport_document.jpg']['file_status'] == 'missing'
    assert final_library['records']['recovered/passport_document.jpg']['sha256'] == original_sha256
    assert exact_copy_path.exists()
    search_after_delete = client.post('/search', json={
        'folder_path': str(library),
        'query': 'government passport identity document',
        'limit': 5,
    })
    assert search_after_delete.status_code == 200
    assert all(
        result['relative_path'] != 'recovered/passport_document.jpg'
        for result in search_after_delete.json()['results']
    )
    assert hashlib.sha256(exact_copy_path.read_bytes()).hexdigest() != original_sha256
    print(f'PHASE9_REAL_INTEGRATION_TIMINGS_SECONDS {phase_timings}')
