import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.services import library_service
from app.services.hashing_service import compute_phash_value, compute_sha256, is_supported_image
from app.services.library_service import scan_library, validate_folder_path

from app.services.metadata_service import extract_image_metadata

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolate_index(tmp_path, monkeypatch):
    index_path = tmp_path / 'data' / 'index' / 'library_index.json'
    monkeypatch.setattr(library_service, 'INDEX_PATH', index_path)
    monkeypatch.setattr(library_service, 'INDEX_DIR', index_path.parent)


def create_image(path: Path, size=(50, 40), color=(12, 34, 56)):
    image = Image.new('RGB', size=size, color=color)
    image.save(path)
    return path


def test_health_endpoint():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok', 'service': 'MemoryOS backend'}


def test_validate_folder_path_rejects_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_folder_path(str(tmp_path / 'missing-folder'))


def test_supported_image_detection():
    assert is_supported_image('sample.jpg') is True
    assert is_supported_image('sample.PNG') is True
    assert is_supported_image('notes.txt') is False


def test_unsupported_file_filtering(tmp_path):
    library_dir = tmp_path / 'library'
    library_dir.mkdir()
    create_image(library_dir / 'one.jpg')
    (library_dir / 'notes.txt').write_text('not an image', encoding='utf-8')

    result = scan_library(str(library_dir))

    assert result['total_discovered_images'] == 1
    assert result['image_records'][0]['filename'] == 'one.jpg'


def test_sha256_generation(tmp_path):
    image_path = create_image(tmp_path / 'hash-test.jpg')
    expected = hashlib.sha256(image_path.read_bytes()).hexdigest()
    assert compute_sha256(str(image_path)) == expected


def test_phash_generation(tmp_path):
    image_path = create_image(tmp_path / 'phash-test.jpg')
    phash_value = compute_phash_value(str(image_path))
    assert isinstance(phash_value, str)
    assert len(phash_value) > 0


def test_metadata_extraction(tmp_path):
    image_path = create_image(tmp_path / 'meta.jpg', size=(100, 50), color=(255, 128, 0))
    metadata = extract_image_metadata(str(image_path), 'meta.jpg')

    assert metadata['filename'] == 'meta.jpg'
    assert metadata['width'] == 100
    assert metadata['height'] == 50
    assert metadata['aspect_ratio'] == 2.0
    assert metadata['processing_status'] == 'pending'
    assert metadata['sha256']
    assert metadata['phash']


def test_first_scan_creates_index_records(tmp_path):
    library_dir = tmp_path / 'library'
    library_dir.mkdir()
    nested_dir = library_dir / 'nested'
    nested_dir.mkdir()
    create_image(library_dir / 'a.jpg')
    create_image(nested_dir / 'b.png')

    result = scan_library(str(library_dir))

    assert result['new_images'] == 2
    assert result['updated_images'] == 0
    assert result['unchanged_images'] == 0
    assert len(result['image_records']) == 2


def test_second_scan_does_not_duplicate_unchanged_files(tmp_path):
    library_dir = tmp_path / 'library'
    library_dir.mkdir()
    create_image(library_dir / 'a.jpg')

    first_result = scan_library(str(library_dir))
    second_result = scan_library(str(library_dir))

    assert first_result['new_images'] == 1
    assert second_result['total_discovered_images'] == 1
    assert second_result['unchanged_images'] == 1
    assert second_result['new_images'] == 0
    assert len(second_result['image_records']) == 1


def test_modified_file_is_detected(tmp_path):
    library_dir = tmp_path / 'library'
    library_dir.mkdir()
    image_path = create_image(library_dir / 'modified.jpg', color=(10, 10, 10))

    scan_library(str(library_dir))
    image = Image.new('RGB', size=(80, 60), color=(200, 200, 200))
    image.save(image_path)

    result = scan_library(str(library_dir))

    assert result['updated_images'] >= 1
    assert any(record['relative_path'] == 'modified.jpg' and record['file_status'] == 'modified' for record in result['image_records'])


def test_index_survives_reload(tmp_path):
    library_dir = tmp_path / 'library'
    library_dir.mkdir()
    create_image(library_dir / 'index-check.jpg')

    scan_library(str(library_dir))

    assert str(library_service.INDEX_PATH).replace('\\', '/').endswith('data/index/library_index.json')
    saved_index = json.loads(library_service.INDEX_PATH.read_text(encoding='utf-8'))

    assert 'libraries' in saved_index
    assert len(saved_index['libraries']) >= 1


def test_original_image_bytes_are_not_copied_into_data_storage(tmp_path):
    library_dir = tmp_path / 'library'
    library_dir.mkdir()
    create_image(library_dir / 'copy-check.jpg')

    scan_library(str(library_dir))
    index_data = json.loads(library_service.INDEX_PATH.read_text(encoding='utf-8'))
    library_record = next(iter(index_data['libraries'].values()))['records']['copy-check.jpg']

    assert 'image_bytes' not in library_record
    assert 'base64' not in json.dumps(library_record).lower()
    assert 'data:' not in json.dumps(library_record).lower()


def test_library_scan_handles_file_instead_of_directory(tmp_path):
    file_path = tmp_path / 'not-a-folder.txt'
    file_path.write_text('content', encoding='utf-8')

    with pytest.raises(ValueError):
        validate_folder_path(str(file_path))
