import hashlib
import json
from pathlib import Path
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import file_operations_service as file_service

client = TestClient(app)


@pytest.fixture
def file_index(tmp_path, monkeypatch):
    index_path = tmp_path / 'data' / 'index' / 'library_index.json'
    monkeypatch.setattr(file_service, 'INDEX_PATH', index_path)
    return index_path


def make_library(index_path: Path, tmp_path: Path, files: dict[str, bytes] | None = None):
    root = tmp_path / 'library'
    root.mkdir(exist_ok=True)
    for relative_path, content in (files or {'folder/photo.png': b'photo-bytes'}).items():
        target = root / Path(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    records = {}
    for relative_path in (files or {'folder/photo.png': b'photo-bytes'}):
        image_path = root / Path(relative_path)
        content = image_path.read_bytes()
        records[relative_path] = {
            'record_id': relative_path,
            'filename': image_path.name,
            'relative_path': relative_path.replace('\\', '/'),
            'extension': image_path.suffix.lower(),
            'file_size': len(content),
            'modified_time': datetime.fromtimestamp(
                image_path.stat().st_mtime
            ).isoformat(timespec='seconds'),
            'sha256': hashlib.sha256(content).hexdigest(),
            'phash': '0123456789abcdef',
            'file_status': 'available',
            'processing_status': 'pending',
            'ai_processing_status': 'completed',
            'ocr_text': 'stored OCR',
            'caption': 'stored caption',
            'embedding_path': 'data/embeddings/existing.npy',
            'embedding_status': 'completed',
            'category': 'Education',
            'classification_status': 'classified',
            'classification_confidence': 0.8,
            'duplicate_group_id': 'old-group',
            'duplicate_type': 'exact',
            'duplicate_groups': [{'group_id': 'old-group', 'duplicate_type': 'exact'}],
        }
    library_id = 'library-test'
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
    return root, library_id, records


def read_library(index_path: Path, library_id: str) -> dict:
    payload = json.loads(index_path.read_text(encoding='utf-8'))
    return payload['libraries'][library_id]


def request_payload(library_id='library-test', record_id='folder/photo.png', **extra):
    return {'library_id': library_id, 'record_id': record_id, **extra}


def test_open_and_open_folder_are_indexed_only_and_do_not_write_index(
    file_index, tmp_path, monkeypatch,
):
    root, library_id, _ = make_library(file_index, tmp_path)
    before = file_index.read_bytes()
    opened = []
    monkeypatch.setattr(file_service, '_open_native', opened.append)

    image_response = client.post('/files/open', json=request_payload(library_id))
    folder_response = client.post('/files/open-folder', json=request_payload(library_id))
    arbitrary = client.post('/files/open', json=request_payload(library_id, 'C:\\outside.png'))

    assert image_response.status_code == 200
    assert folder_response.status_code == 200
    assert arbitrary.status_code == 400
    assert opened == [root / 'folder' / 'photo.png', root / 'folder']
    assert file_index.read_bytes() == before


def test_preview_serves_only_an_indexed_local_image_without_changing_index(
    file_index, tmp_path,
):
    root, library_id, _ = make_library(file_index, tmp_path)
    before = file_index.read_bytes()

    response = client.get('/files/preview', params={
        'library_id': library_id,
        'record_id': 'folder/photo.png',
    })
    arbitrary = client.get('/files/preview', params={
        'library_id': library_id,
        'record_id': str(tmp_path / 'outside.png'),
    })

    assert response.status_code == 200
    assert response.content == b'photo-bytes'
    assert response.headers['content-type'] == 'image/png'
    assert response.headers['cache-control'] == 'no-store'
    assert response.headers['x-content-type-options'] == 'nosniff'
    assert arbitrary.status_code == 400
    assert file_index.read_bytes() == before


def test_open_rejects_missing_source_and_invalid_record(file_index, tmp_path, monkeypatch):
    root, library_id, _ = make_library(file_index, tmp_path)
    monkeypatch.setattr(file_service, '_open_native', lambda _path: None)
    (root / 'folder' / 'photo.png').unlink()

    missing = client.post('/files/open', json=request_payload(library_id))
    invalid = client.post('/files/open-folder', json=request_payload(library_id, 'not-indexed.png'))

    assert missing.status_code == 400
    assert invalid.status_code == 400


@pytest.mark.parametrize('unsafe_path', [
    '../library-sibling/escape.png',
    'folder/../../outside.png',
    'C:\\outside.png',
    '\\\\server\\share\\outside.png',
    'folder\\..\\..\\outside.png',
])
def test_indexed_path_traversal_and_sibling_prefix_attacks_are_rejected(
    file_index, tmp_path, unsafe_path,
):
    root, library_id, _ = make_library(file_index, tmp_path)
    outside = tmp_path / 'library-sibling' / 'escape.png'
    outside.parent.mkdir()
    outside.write_bytes(b'not in library')
    payload = json.loads(file_index.read_text(encoding='utf-8'))
    library = payload['libraries'][library_id]
    record = library['records'].pop('folder/photo.png')
    record['relative_path'] = unsafe_path
    library['records'][unsafe_path] = record
    file_index.write_text(json.dumps(payload), encoding='utf-8')
    before = file_index.read_bytes()

    response = client.post('/files/delete', json=request_payload(
        library_id, unsafe_path, confirm=True,
    ))

    assert response.status_code == 400
    assert outside.read_bytes() == b'not in library'
    assert (root / 'folder' / 'photo.png').exists()
    assert file_index.read_bytes() == before


def test_symlink_source_is_rejected_when_supported(file_index, tmp_path):
    root, library_id, _ = make_library(file_index, tmp_path)
    external = tmp_path / 'outside.png'
    external.write_bytes(b'outside')
    source = root / 'folder' / 'photo.png'
    source.unlink()
    try:
        source.symlink_to(external)
    except (OSError, NotImplementedError):
        pytest.skip('Symlink creation is not available for this account.')

    response = client.post('/files/delete', json=request_payload(
        library_id, confirm=True,
    ))

    assert response.status_code == 400
    assert external.read_bytes() == b'outside'


def test_rename_updates_only_path_metadata_and_preserves_ai_artifacts(file_index, tmp_path):
    root, library_id, records = make_library(file_index, tmp_path)
    source_bytes = (root / 'folder' / 'photo.png').read_bytes()
    original = records['folder/photo.png']

    response = client.post('/files/rename', json=request_payload(
        library_id, new_filename='renamed',
    ))

    assert response.status_code == 200
    destination = root / 'folder' / 'renamed.png'
    assert not (root / 'folder' / 'photo.png').exists()
    assert destination.read_bytes() == source_bytes
    record = read_library(file_index, library_id)['records']['folder/renamed.png']
    assert record['record_id'] == record['relative_path'] == 'folder/renamed.png'
    assert record['filename'] == 'renamed.png'
    assert record['extension'] == '.png'
    for key in ('sha256', 'phash', 'ai_processing_status', 'ocr_text', 'caption',
                'embedding_path', 'embedding_status', 'category',
                'classification_status', 'classification_confidence'):
        assert record[key] == original[key]
    assert 'duplicate_detection' not in read_library(file_index, library_id)
    assert 'duplicate_groups' not in record


@pytest.mark.parametrize('filename', [
    '', '.', '..', '../escape.png', 'folder\\escape.png', 'folder/escape.png',
    'C:\\outside.png', '\\\\server\\share\\outside.png', 'bad?.png',
    'CON.png', 'trailing. ',
])
def test_rename_rejects_invalid_names_without_changing_file_or_index(
    file_index, tmp_path, filename,
):
    root, library_id, _ = make_library(file_index, tmp_path)
    before = file_index.read_bytes()

    response = client.post('/files/rename', json=request_payload(
        library_id, new_filename=filename,
    ))

    assert response.status_code in (400, 422)
    assert (root / 'folder' / 'photo.png').exists()
    assert file_index.read_bytes() == before


def test_rename_allows_only_supported_explicit_extensions_and_rejects_existing_target(
    file_index, tmp_path,
):
    root, library_id, _ = make_library(file_index, tmp_path)
    same_extension_change = client.post('/files/rename', json=request_payload(
        library_id, new_filename='renamed.jpg',
    ))
    assert same_extension_change.status_code == 200

    (root / 'folder' / 'existing.png').write_bytes(b'existing')
    existing = client.post('/files/rename', json=request_payload(
        library_id, 'folder/renamed.jpg', new_filename='existing.png',
    ))
    unsupported = client.post('/files/rename', json=request_payload(
        library_id, 'folder/renamed.jpg', new_filename='renamed.gif.exe',
    ))
    assert existing.status_code == 409
    assert unsupported.status_code == 400
    assert (root / 'folder' / 'existing.png').read_bytes() == b'existing'


def test_rename_rejects_missing_or_changed_source(file_index, tmp_path):
    root, library_id, _ = make_library(file_index, tmp_path)
    (root / 'folder' / 'photo.png').write_bytes(b'modified bytes')
    before = file_index.read_bytes()

    changed = client.post('/files/rename', json=request_payload(
        library_id, new_filename='renamed.png',
    ))
    assert changed.status_code == 409
    assert (root / 'folder' / 'photo.png').exists()
    assert file_index.read_bytes() == before

    (root / 'folder' / 'photo.png').unlink()
    missing = client.post('/files/rename', json=request_payload(
        library_id, new_filename='renamed.png',
    ))
    assert missing.status_code == 400
    assert file_index.read_bytes() == before


def test_move_uses_library_relative_existing_folder_and_preserves_artifacts(file_index, tmp_path):
    root, library_id, records = make_library(file_index, tmp_path)
    (root / 'destination').mkdir()
    source_bytes = (root / 'folder' / 'photo.png').read_bytes()
    original = records['folder/photo.png']

    response = client.post('/files/move', json=request_payload(
        library_id, destination_folder='destination',
    ))

    assert response.status_code == 200
    assert not (root / 'folder' / 'photo.png').exists()
    assert (root / 'destination' / 'photo.png').read_bytes() == source_bytes
    record = read_library(file_index, library_id)['records']['destination/photo.png']
    assert record['sha256'] == original['sha256']
    assert record['phash'] == original['phash']
    assert record['ai_processing_status'] == 'completed'
    assert record['ocr_text'] == 'stored OCR'
    assert record['caption'] == 'stored caption'
    assert record['embedding_path'] == 'data/embeddings/existing.npy'
    assert record['category'] == 'Education'


@pytest.mark.parametrize('destination', [
    '../outside', '..\\outside', 'folder/../../outside', 'C:\\outside',
    '\\\\server\\share', '//server/share', 'destination\\..\\outside',
])
def test_move_rejects_traversal_and_absolute_destination(file_index, tmp_path, destination):
    root, library_id, _ = make_library(file_index, tmp_path)
    before = file_index.read_bytes()

    response = client.post('/files/move', json=request_payload(
        library_id, destination_folder=destination,
    ))

    assert response.status_code == 400
    assert (root / 'folder' / 'photo.png').exists()
    assert file_index.read_bytes() == before


def test_move_rejects_missing_folder_and_existing_destination(file_index, tmp_path):
    root, library_id, _ = make_library(file_index, tmp_path)
    missing_folder = client.post('/files/move', json=request_payload(
        library_id, destination_folder='not-here',
    ))
    assert missing_folder.status_code == 400
    (root / 'destination').mkdir()
    (root / 'destination' / 'photo.png').write_bytes(b'existing destination')
    existing = client.post('/files/move', json=request_payload(
        library_id, destination_folder='destination',
    ))
    assert existing.status_code == 409
    assert (root / 'folder' / 'photo.png').exists()
    assert (root / 'destination' / 'photo.png').read_bytes() == b'existing destination'


def test_move_rejects_modified_source(file_index, tmp_path):
    root, library_id, _ = make_library(file_index, tmp_path)
    (root / 'destination').mkdir()
    (root / 'folder' / 'photo.png').write_bytes(b'changed')

    response = client.post('/files/move', json=request_payload(
        library_id, destination_folder='destination',
    ))

    assert response.status_code == 409
    assert (root / 'folder' / 'photo.png').exists()


def test_filesystem_move_failure_does_not_update_index(file_index, tmp_path, monkeypatch):
    root, library_id, _ = make_library(file_index, tmp_path)
    (root / 'destination').mkdir()
    before = file_index.read_bytes()
    monkeypatch.setattr(
        file_service,
        '_move_no_overwrite',
        lambda *_args: (_ for _ in ()).throw(
            file_service.FileOperationError('simulated filesystem failure')
        ),
    )

    response = client.post('/files/move', json=request_payload(
        library_id, destination_folder='destination',
    ))

    assert response.status_code == 400
    assert file_index.read_bytes() == before
    assert (root / 'folder' / 'photo.png').exists()


@pytest.mark.parametrize('confirm', [False, None])
def test_delete_without_explicit_confirmation_does_nothing(
    file_index, tmp_path, confirm,
):
    root, library_id, _ = make_library(file_index, tmp_path)
    before = file_index.read_bytes()
    payload = request_payload(library_id)
    if confirm is not None:
        payload['confirm'] = confirm

    response = client.post('/files/delete', json=payload)

    assert response.status_code in (400, 422)
    assert (root / 'folder' / 'photo.png').exists()
    assert file_index.read_bytes() == before


def test_confirmed_delete_marks_record_missing_and_never_deletes_group_automatically(
    file_index, tmp_path,
):
    root, library_id, _ = make_library(file_index, tmp_path, {
        'folder/photo.png': b'photo-bytes',
        'folder/duplicate.png': b'other-bytes',
    })
    payload = json.loads(file_index.read_text(encoding='utf-8'))
    library = payload['libraries'][library_id]
    library['duplicate_detection'] = {
        'groups': [{
            'group_id': 'duplicate',
            'duplicate_type': 'exact',
            'members': [
                {'relative_path': 'folder/photo.png'},
                {'relative_path': 'folder/duplicate.png'},
            ],
        }],
    }
    file_index.write_text(json.dumps(payload), encoding='utf-8')

    response = client.post('/files/delete', json=request_payload(
        library_id, confirm=True,
    ))

    assert response.status_code == 200
    assert not (root / 'folder' / 'photo.png').exists()
    assert (root / 'folder' / 'duplicate.png').exists()
    saved = read_library(file_index, library_id)
    assert saved['records']['folder/photo.png']['file_status'] == 'missing'
    assert 'duplicate_detection' not in saved
    assert 'duplicate_groups' not in saved['records']['folder/duplicate.png']


def test_delete_rejects_directory_even_if_indexed_as_image(file_index, tmp_path):
    root, library_id, _ = make_library(file_index, tmp_path)
    image_path = root / 'folder' / 'photo.png'
    image_path.unlink()
    image_path.mkdir()

    response = client.post('/files/delete', json=request_payload(
        library_id, confirm=True,
    ))

    assert response.status_code == 400
    assert image_path.is_dir()


def test_file_marked_missing_cannot_be_operated_on_until_rescanned(file_index, tmp_path):
    root, library_id, _ = make_library(file_index, tmp_path)
    payload = json.loads(file_index.read_text(encoding='utf-8'))
    payload['libraries'][library_id]['records']['folder/photo.png']['file_status'] = 'missing'
    file_index.write_text(json.dumps(payload), encoding='utf-8')
    original = root / 'folder' / 'photo.png'

    response = client.post('/files/rename', json=request_payload(
        library_id, new_filename='renamed.png',
    ))

    assert response.status_code == 400
    assert original.exists()


def test_failed_index_save_reports_reconciliation_without_rolling_back_rename(
    file_index, tmp_path, monkeypatch,
):
    root, library_id, _ = make_library(file_index, tmp_path)
    monkeypatch.setattr(
        file_service.os,
        'fsync',
        lambda _descriptor: (_ for _ in ()).throw(OSError('simulated save failure')),
    )

    response = client.post('/files/rename', json=request_payload(
        library_id, new_filename='renamed.png',
    ))

    assert response.status_code == 500
    assert 'Rescan the library' in response.json()['detail']
    assert (root / 'folder' / 'renamed.png').exists()
    assert not (root / 'folder' / 'photo.png').exists()


def test_folder_picker_only_lists_local_non_symlink_library_folders(file_index, tmp_path):
    root, library_id, _ = make_library(file_index, tmp_path)
    (root / 'destination').mkdir()
    response = client.get('/files/folders', params={
        'library_id': library_id,
        'record_id': 'folder/photo.png',
    })

    assert response.status_code == 200
    assert set(response.json()['folders']) == {'', 'destination', 'folder'}
