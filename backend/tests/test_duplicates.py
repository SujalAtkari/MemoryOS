import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

from app.main import app
from app.services import duplicate_service, library_service
from app.services.duplicate_service import (
    DUPLICATE_VERSION,
    _build_visual_edges,
    _phash_distance,
    process_library_duplicates,
)

client = TestClient(app)


@pytest.fixture
def duplicate_index(tmp_path, monkeypatch):
    index_path = tmp_path / 'data' / 'index' / 'library_index.json'
    monkeypatch.setattr(duplicate_service, 'INDEX_PATH', index_path)
    monkeypatch.setattr(library_service, 'INDEX_PATH', index_path)
    monkeypatch.setattr(library_service, 'INDEX_DIR', index_path.parent)
    return index_path


def make_record(library: Path, name: str, sha: str, phash: str, vector=None, **extra) -> dict:
    image_path = library / name
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(name.encode('utf-8'))
    embedding_path = library.parent / 'embeddings' / f'{Path(name).stem}.npy'
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    if vector is not None:
        np.save(embedding_path, np.asarray(vector, dtype=np.float32))
    return {
        'record_id': name,
        'filename': Path(name).name,
        'relative_path': name,
        'sha256': sha,
        'phash': phash,
        'file_status': 'available',
        'processing_status': 'pending',
        'embedding_status': 'completed' if vector is not None else 'failed',
        'embedding_path': str(embedding_path) if vector is not None else '',
        **extra,
    }


def write_index(path: Path, library: Path, records: dict) -> str:
    library_id = library_service._library_id_for_path(str(library.resolve()))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        'version': 1,
        'libraries': {
            library_id: {
                'library_root': str(library.resolve()),
                'records': records,
            }
        },
    }), encoding='utf-8')
    return library_id


def test_identical_sha256_groups_two_and_multiple_records_without_openclip(tmp_path, duplicate_index, monkeypatch):
    library = tmp_path / 'library'
    library.mkdir()
    records = {
        'one.jpg': make_record(library, 'one.jpg', 'a' * 64, '0000000000000000'),
        'copy.jpg': make_record(library, 'copy.jpg', 'a' * 64, '0000000000000000'),
        'backup/one.jpg': make_record(library, 'backup/one.jpg', 'a' * 64, '0000000000000000'),
        'different.jpg': make_record(library, 'different.jpg', 'b' * 64, 'ffffffffffffffff'),
    }
    write_index(duplicate_index, library, records)
    monkeypatch.setattr(
        duplicate_service,
        '_record_embedding',
        lambda *_: pytest.fail('Exact duplicates should not require OpenCLIP embeddings'),
    )

    result = process_library_duplicates(str(library), include_visual=False)

    assert result['exact_groups'] == 1
    assert len(result['groups'][0]['members']) == 3
    assert result['groups'][0]['match_method'] == 'sha256'
    assert result['groups'][0]['duplicate_confidence'] == 1.0


def test_different_sha256_values_are_not_exact_duplicates(tmp_path, duplicate_index):
    library = tmp_path / 'library'
    library.mkdir()
    records = {
        'one.jpg': make_record(library, 'one.jpg', 'a' * 64, '0000000000000000'),
        'two.jpg': make_record(library, 'two.jpg', 'b' * 64, 'ffffffffffffffff'),
    }
    write_index(duplicate_index, library, records)

    result = process_library_duplicates(str(library), include_visual=False)

    assert result['exact_groups'] == 0
    assert result['duplicates_found'] == 0


def test_phash_near_duplicate_match_and_nonmatch(tmp_path, duplicate_index):
    library = tmp_path / 'library'
    library.mkdir()
    records = {
        'base.jpg': make_record(library, 'base.jpg', 'a' * 64, '0000000000000000'),
        'near.jpg': make_record(library, 'near.jpg', 'b' * 64, '0000000000000007'),
        'far.jpg': make_record(library, 'far.jpg', 'c' * 64, '0000000000003fff'),
    }
    write_index(duplicate_index, library, records)

    result = process_library_duplicates(str(library), include_visual=False, phash_threshold=8)

    assert result['near_duplicate_groups'] == 1
    near_group = next(group for group in result['groups'] if group['duplicate_type'] == 'near')
    assert {member['relative_path'] for member in near_group['members']} == {'base.jpg', 'near.jpg'}
    assert near_group['matches'][0]['distance'] == 3
    assert _phash_distance('not-a-hash', '0000000000000000') is None


@pytest.mark.parametrize(
    'phash',
    [None, '', 'invalid', 'xyz', '00000000000000000'],
)
def test_missing_or_invalid_phash_is_ignored(tmp_path, duplicate_index, phash):
    library = tmp_path / 'library'
    library.mkdir()
    records = {
        'invalid.jpg': make_record(library, 'invalid.jpg', 'a' * 64, phash),
        'valid.jpg': make_record(library, 'valid.jpg', 'b' * 64, '0000000000000000'),
    }
    write_index(duplicate_index, library, records)

    result = process_library_duplicates(str(library), include_visual=False)

    assert result['near_duplicate_groups'] == 0


def test_openclip_visual_match_for_non_exact_non_phash_pair(monkeypatch):
    records = {
        'left.jpg': {'sha256': 'a' * 64, 'embedding_path': 'left.npy', 'embedding_status': 'completed'},
        'right.jpg': {'sha256': 'b' * 64, 'embedding_path': 'right.npy', 'embedding_status': 'completed'},
    }
    vectors = {
        'left.npy': np.array([1.0, 0.0], dtype=np.float32),
        'right.npy': np.array([0.95, 0.3122499], dtype=np.float32),
    }

    monkeypatch.setattr(
        duplicate_service,
        '_record_embedding',
        lambda record: vectors[record['embedding_path']],
    )
    edges = _build_visual_edges(records, {('left.jpg', 'right.jpg')}, 0.90, set())
    assert len(edges) == 1
    assert edges[0]['match_method'] == 'openclip'
    assert edges[0]['similarity'] == pytest.approx(0.95, abs=1e-5)


def test_below_threshold_visual_pair_is_not_reported(monkeypatch):
    records = {
        'left.jpg': {'sha256': 'a' * 64, 'embedding_path': 'left.npy', 'embedding_status': 'completed'},
        'right.jpg': {'sha256': 'b' * 64, 'embedding_path': 'right.npy', 'embedding_status': 'completed'},
    }
    vectors = {
        'left.npy': np.array([1.0, 0.0], dtype=np.float32),
        'right.npy': np.array([0.8, 0.6], dtype=np.float32),
    }
    monkeypatch.setattr(
        duplicate_service,
        '_record_embedding',
        lambda record: vectors[record['embedding_path']],
    )
    edges = _build_visual_edges(records, {('left.jpg', 'right.jpg')}, 0.90, set())

    assert edges == []


def test_visual_matching_skips_incompatible_embedding_dimensions_without_crashing(monkeypatch):
    records = {
        path: {
            'sha256': hashlib.sha256(path.encode('utf-8')).hexdigest(),
            'embedding_path': path,
            'embedding_status': 'completed',
        }
        for path in ('left-2.jpg', 'right-2.jpg', 'left-3.jpg', 'right-3.jpg')
    }
    vectors = {
        'left-2.jpg': np.array([1.0, 0.0], dtype=np.float32),
        'right-2.jpg': np.array([0.95, 0.3122499], dtype=np.float32),
        'left-3.jpg': np.array([1.0, 0.0, 0.0], dtype=np.float32),
        'right-3.jpg': np.array([0.95, 0.3122499, 0.0], dtype=np.float32),
    }
    monkeypatch.setattr(
        duplicate_service,
        '_record_embedding',
        lambda record: vectors[record['embedding_path']],
    )
    candidate_pairs = {
        ('left-2.jpg', 'right-2.jpg'),
        ('left-3.jpg', 'right-3.jpg'),
    }

    edges = _build_visual_edges(records, candidate_pairs, 0.90, set())

    assert len(edges) == 2
    assert {edge['match_method'] for edge in edges} == {'openclip'}


def test_missing_and_invalid_embeddings_do_not_crash(tmp_path, duplicate_index):
    library = tmp_path / 'library'
    library.mkdir()
    records = {
        'one.jpg': make_record(library, 'one.jpg', 'a' * 64, '0000000000000000'),
        'missing.jpg': make_record(library, 'missing.jpg', 'b' * 64, '00000000000003ff'),
        'invalid.jpg': make_record(library, 'invalid.jpg', 'c' * 64, '00000000000007ff'),
    }
    invalid_path = library.parent / 'embeddings' / 'invalid.npy'
    invalid_path.write_bytes(b'broken npy')
    records['missing.jpg']['embedding_status'] = 'completed'
    records['missing.jpg']['embedding_path'] = str(library.parent / 'missing.npy')
    records['invalid.jpg']['embedding_status'] = 'completed'
    records['invalid.jpg']['embedding_path'] = str(invalid_path)
    write_index(duplicate_index, library, records)

    result = process_library_duplicates(str(library), include_visual=True)

    assert result['processed'] == 3
    assert result['visual_matches'] == 0


def test_duplicate_processing_skips_unchanged_library_and_group_ids_are_stable(tmp_path, duplicate_index):
    library = tmp_path / 'library'
    library.mkdir()
    records = {
        'one.jpg': make_record(library, 'one.jpg', 'a' * 64, '0000000000000000'),
        'two.jpg': make_record(library, 'two.jpg', 'a' * 64, '0000000000000000'),
    }
    write_index(duplicate_index, library, records)

    first = process_library_duplicates(str(library), include_visual=False)
    second = process_library_duplicates(str(library), include_visual=False)
    forced = process_library_duplicates(str(library), include_visual=False, force=True)

    assert first['groups'][0]['group_id'] == second['groups'][0]['group_id'] == forced['groups'][0]['group_id']
    assert first['processed'] == 2
    assert second['processed'] == 0
    assert second['skipped'] == 2
    assert forced['processed'] == 2


def test_duplicate_fields_preserve_phase_one_to_four_metadata_and_never_delete(
    tmp_path,
    duplicate_index,
):
    library = tmp_path / 'library'
    library.mkdir()
    records = {
        'one.jpg': make_record(
            library, 'one.jpg', 'a' * 64, '0000000000000000',
            ocr_text='invoice', caption='receipt', embedding_path='unused',
            category='Bills & Receipts', classification_status='classified',
            classification_confidence=0.9, ai_processing_status='completed',
        ),
        'two.jpg': make_record(
            library, 'two.jpg', 'a' * 64, '0000000000000000',
            ocr_text='invoice', caption='receipt', category='Bills & Receipts',
            classification_status='classified', classification_confidence=0.9,
            ai_processing_status='completed',
        ),
    }
    write_index(duplicate_index, library, records)
    before_hashes = {path: hashlib.sha256((library / path).read_bytes()).hexdigest() for path in records}

    process_library_duplicates(str(library), include_visual=False)

    payload = json.loads(duplicate_index.read_text(encoding='utf-8'))
    saved = next(iter(payload['libraries'].values()))['records']
    for relative_path in records:
        assert saved[relative_path]['ocr_text'] == 'invoice'
        assert saved[relative_path]['caption'] == 'receipt'
        assert saved[relative_path]['category'] == 'Bills & Receipts'
        assert saved[relative_path]['classification_status'] == 'classified'
        assert saved[relative_path]['ai_processing_status'] == 'completed'
        assert hashlib.sha256((library / relative_path).read_bytes()).hexdigest() == before_hashes[relative_path]
    assert all((library / relative_path).exists() for relative_path in records)
    assert not (tmp_path / 'data' / 'one.jpg').exists()


def test_missing_files_are_skipped_safely(tmp_path, duplicate_index):
    library = tmp_path / 'library'
    library.mkdir()
    records = {
        'gone.jpg': {
            'record_id': 'gone.jpg',
            'relative_path': 'gone.jpg',
            'filename': 'gone.jpg',
            'sha256': 'a' * 64,
            'phash': '0000000000000000',
            'file_status': 'missing',
        }
    }
    write_index(duplicate_index, library, records)

    result = process_library_duplicates(str(library), include_visual=False)

    assert result['total_records'] == 1
    assert result['processed'] == 0
    assert result['duplicates_found'] == 0


def test_large_synthetic_vectorized_visual_candidate_set(monkeypatch):
    count = 400
    records = {
        f'{index:04}.jpg': {
            'sha256': f'{index:064x}',
            'embedding_path': f'{index}.npy',
            'embedding_status': 'completed',
        }
        for index in range(count)
    }
    vectors = {
        f'{index}.npy': np.array([1.0, index / count], dtype=np.float32)
        for index in range(count)
    }
    vectors = {
        key: value / np.linalg.norm(value)
        for key, value in vectors.items()
    }
    monkeypatch.setattr(duplicate_service, '_record_embedding', lambda record: vectors[record['embedding_path']])
    pairs = {(f'{index:04}.jpg', f'{index + 1:04}.jpg') for index in range(count - 1)}

    edges = _build_visual_edges(records, pairs, threshold=0.90, near_pairs=set())

    assert len(edges) == count - 1


def test_duplicate_api_process_and_list(tmp_path, duplicate_index):
    library = tmp_path / 'library'
    library.mkdir()
    records = {
        'one.jpg': make_record(library, 'one.jpg', 'a' * 64, '0000000000000000'),
        'two.jpg': make_record(library, 'two.jpg', 'a' * 64, '0000000000000000'),
    }
    write_index(duplicate_index, library, records)

    process_response = client.post('/duplicates/process', json={
        'folder_path': str(library),
        'include_visual': False,
    })
    list_response = client.get('/duplicates', params={'folder_path': str(library)})

    assert process_response.status_code == 200
    assert process_response.json()['exact_groups'] == 1
    assert list_response.status_code == 200
    assert list_response.json()['groups'][0]['duplicate_type'] == 'exact'
    assert list_response.json()['groups'][0]['members'][0]['relative_path'] in records
    assert DUPLICATE_VERSION == 'phase5-duplicates-v1'


def test_visual_candidate_and_thresholds_are_configurable(tmp_path, duplicate_index, monkeypatch):
    library = tmp_path / 'library'
    library.mkdir()
    records = {
        'one.jpg': make_record(library, 'one.jpg', 'a' * 64, '0000000000000000', [1.0, 0.0]),
        'two.jpg': make_record(library, 'two.jpg', 'b' * 64, '00000000000003ff', [0.95, 0.3122499]),
    }
    write_index(duplicate_index, library, records)
    vectors = {
        record['embedding_path']: np.load(record['embedding_path'])
        for record in records.values()
    }
    vectors = {key: value / np.linalg.norm(value) for key, value in vectors.items()}
    monkeypatch.setattr(duplicate_service, '_record_embedding', lambda record: vectors[record['embedding_path']])

    result = process_library_duplicates(
        str(library),
        include_visual=True,
        phash_threshold=8,
        openclip_threshold=0.90,
        visual_candidate_phash_distance=16,
    )

    assert result['visual_matches'] == 1
    visual = next(group for group in result['groups'] if group['duplicate_type'] == 'visual')
    assert visual['matches'][0]['similarity'] == pytest.approx(0.95, abs=1e-5)


def test_scan_to_duplicate_detection_preserves_existing_ai_artifacts(tmp_path, duplicate_index):
    library = tmp_path / 'library'
    library.mkdir()
    base = Image.new('RGB', (128, 128), 'white')
    draw = ImageDraw.Draw(base)
    draw.rectangle((16, 16, 112, 112), fill='navy')
    draw.rectangle((40, 40, 88, 88), fill='gold')
    base_path = library / 'base.png'
    base.save(base_path)

    exact_path = library / 'exact-copy.png'
    exact_path.write_bytes(base_path.read_bytes())

    near = base.copy()
    ImageDraw.Draw(near).point((20, 20), fill='red')
    near_path = library / 'near.png'
    near.save(near_path)

    unrelated_pixels = np.random.default_rng(17).integers(
        0, 256, size=(128, 128, 3), dtype=np.uint8
    )
    Image.fromarray(unrelated_pixels).save(library / 'unrelated.png')

    scan_result = library_service.scan_library(str(library))
    payload = json.loads(duplicate_index.read_text(encoding='utf-8'))
    library_index = payload['libraries'][scan_result['library_id']]
    artifact_snapshots = {}
    for number, record in enumerate(library_index['records'].values()):
        record.update({
            'ocr_text': f'preserved OCR {number}',
            'caption': f'preserved caption {number}',
            'ai_processing_status': 'completed',
            'category': 'Notes & Documents',
            'classification_status': 'classified',
            'classification_confidence': 0.75,
        })
        embedding_path = tmp_path / 'data' / 'embeddings' / f'{number}.npy'
        embedding_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(embedding_path, np.array([1.0, float(number + 1)], dtype=np.float32))
        record['embedding_path'] = str(embedding_path)
        record['embedding_status'] = 'completed'
        artifact_snapshots[record['relative_path']] = (
            {key: record[key] for key in (
                'ocr_text', 'caption', 'ai_processing_status', 'category',
                'classification_status', 'classification_confidence',
            )},
            embedding_path.read_bytes(),
        )
    duplicate_index.write_text(json.dumps(payload), encoding='utf-8')
    original_bytes = {
        path.name: path.read_bytes()
        for path in library.iterdir()
        if path.is_file()
    }

    response = client.post('/duplicates/process', json={
        'folder_path': str(library),
        'include_visual': False,
    })
    repeat_response = client.post('/duplicates/process', json={
        'folder_path': str(library),
        'include_visual': False,
    })

    assert response.status_code == 200
    result = response.json()
    assert result['exact_groups'] == 1
    assert result['near_duplicate_groups'] >= 1
    assert result['processed'] == 4
    assert repeat_response.status_code == 200
    assert repeat_response.json()['processed'] == 0
    assert repeat_response.json()['skipped'] == 4

    saved = json.loads(duplicate_index.read_text(encoding='utf-8'))
    saved_records = saved['libraries'][scan_result['library_id']]['records']
    for relative_path, (metadata, embedding_bytes) in artifact_snapshots.items():
        assert {key: saved_records[relative_path][key] for key in metadata} == metadata
        assert Path(saved_records[relative_path]['embedding_path']).read_bytes() == embedding_bytes
    assert {
        path.name: path.read_bytes()
        for path in library.iterdir()
        if path.is_file()
    } == original_bytes
    assert len(list(library.iterdir())) == 4
