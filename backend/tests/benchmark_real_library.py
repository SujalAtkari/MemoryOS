import hashlib
import json
import tempfile
import time
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.main import app
from app.services import (
    ai_processing_service,
    classification_service,
    dashboard_service,
    duplicate_service,
    library_service,
    search_service,
)


IMAGE_COUNT = 10


def _timed(timings: dict[str, float], name: str, operation):
    started = time.perf_counter()
    result = operation()
    timings[name] = round(time.perf_counter() - started, 6)
    return result


def _create_image(path: Path, index: int) -> None:
    image = Image.new(
        'RGB',
        (720, 420),
        color=((index * 37) % 255, (index * 71) % 255, (index * 113) % 255),
    )
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 700, 400), outline='white', width=6)
    draw.text((55, 80), f'PASSPORT DOCUMENT {index:02d}', fill='white')
    draw.text((55, 150), 'GOVERNMENT IDENTITY SAMPLE', fill='white')
    draw.rectangle((520, 220, 660, 360), fill=(255 - index * 10, 80, 30))
    image.save(path, format='JPEG', quality=90)


def main() -> None:
    timings: dict[str, float] = {}
    client = TestClient(app)

    with tempfile.TemporaryDirectory(prefix='memoryos-phase9-real-') as temporary:
        temporary_path = Path(temporary)
        library_path = temporary_path / 'image-library'
        index_path = temporary_path / 'data' / 'index' / 'library_index.json'
        library_path.mkdir()
        for index in range(IMAGE_COUNT):
            _create_image(library_path / f'passport-document-{index:02d}.jpg', index)

        source_hashes = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in library_path.iterdir()
        }
        patched_values = []
        for module, attribute, value in (
            (library_service, 'INDEX_PATH', index_path),
            (library_service, 'INDEX_DIR', index_path.parent),
            (ai_processing_service, 'INDEX_PATH', index_path),
            (ai_processing_service, '_project_root', lambda: temporary_path),
            (classification_service, 'INDEX_PATH', index_path),
            (duplicate_service, 'INDEX_PATH', index_path),
            (dashboard_service, 'INDEX_PATH', index_path),
            (search_service, 'INDEX_PATH', index_path),
        ):
            patched_values.append((module, attribute, getattr(module, attribute)))
            setattr(module, attribute, value)

        try:
            scanned = _timed(
                timings,
                'phase1_scan_10_images_seconds',
                lambda: client.post('/library/scan', json={'folder_path': str(library_path)}),
            )
            assert scanned.status_code == 200, scanned.text
            assert scanned.json()['total_discovered_images'] == IMAGE_COUNT

            processed = _timed(
                timings,
                'phase2_real_ai_process_10_images_seconds',
                lambda: client.post('/ai/process', json={'folder_path': str(library_path)}),
            )
            assert processed.status_code == 200, processed.text
            assert processed.json()['processed'] == IMAGE_COUNT, processed.json()
            assert processed.json()['failed'] == 0, processed.json()

            classified = _timed(
                timings,
                'phase3_classify_10_images_seconds',
                lambda: client.post(
                    '/classification/process',
                    json={'folder_path': str(library_path)},
                ),
            )
            assert classified.status_code == 200, classified.text

            search_result = _timed(
                timings,
                'phase4_search_10_images_seconds',
                lambda: client.post('/search', json={
                    'folder_path': str(library_path),
                    'query': 'government passport identity document',
                    'limit': 10,
                }),
            )
            assert search_result.status_code == 200, search_result.text
            assert search_result.json()['total_results'] > 0, search_result.json()

            duplicates = _timed(
                timings,
                'phase5_duplicates_10_images_seconds',
                lambda: client.post('/duplicates/process', json={
                    'folder_path': str(library_path),
                    'include_visual': False,
                }),
            )
            assert duplicates.status_code == 200, duplicates.text

            dashboard = _timed(
                timings,
                'phase6_dashboard_10_images_seconds',
                lambda: client.get('/dashboard/stats', params={
                    'folder_path': str(library_path),
                }),
            )
            assert dashboard.status_code == 200, dashboard.text
            dashboard_stats = dashboard.json()
            assert dashboard_stats['total_images'] == IMAGE_COUNT
            assert dashboard_stats['processed_images'] == IMAGE_COUNT

            index = json.loads(index_path.read_text(encoding='utf-8'))
            records = next(iter(index['libraries'].values()))['records']
            assert len(records) == IMAGE_COUNT
            assert all(Path(record['embedding_path']).is_file() for record in records.values())
            assert {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in library_path.iterdir()
            } == source_hashes
            copied_images = [
                path for path in (temporary_path / 'data').rglob('*')
                if path.is_file() and path.suffix.lower() in {
                    '.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tiff', '.tif',
                }
            ]
            assert copied_images == []
        finally:
            for module, attribute, original_value in reversed(patched_values):
                setattr(module, attribute, original_value)

        print(json.dumps({
            'dataset': '10 locally generated JPEGs; real local OCR, BLIP, and OpenCLIP inference',
            'image_count': IMAGE_COUNT,
            'ai_processed': processed.json()['processed'],
            'ai_failed': processed.json()['failed'],
            'classified': classified.json().get('classified'),
            'search_results': search_result.json()['total_results'],
            'dashboard_total_images': dashboard_stats['total_images'],
            'dashboard_processed_images': dashboard_stats['processed_images'],
            'exact_duplicate_groups': duplicates.json().get('exact_groups'),
            'near_duplicate_groups': duplicates.json().get('near_duplicate_groups'),
            'phase_timings_seconds': timings,
            'original_files_unchanged': True,
            'original_images_copied_to_data': False,
        }, indent=2))


if __name__ == '__main__':
    main()
