from datetime import datetime
from pathlib import Path

from PIL import Image

from app.services.hashing_service import compute_phash_value, compute_sha256


def extract_image_metadata(file_path: str, relative_path: str) -> dict:
    image_path = Path(file_path)
    file_stats = image_path.stat()

    width = 0
    height = 0
    try:
        with Image.open(image_path) as image:
            width, height = image.size
    except Exception:
        width = 0
        height = 0

    aspect_ratio = round(width / height, 6) if height else 0

    return {
        'filename': image_path.name,
        'relative_path': relative_path.replace('\\', '/'),
        'extension': image_path.suffix.lower(),
        'file_size': file_stats.st_size,
        'modified_time': datetime.fromtimestamp(file_stats.st_mtime).isoformat(timespec='seconds'),
        'width': width,
        'height': height,
        'aspect_ratio': aspect_ratio,
        'sha256': compute_sha256(file_path),
        'phash': compute_phash_value(file_path),
        'processing_status': 'pending',
        'file_status': 'available',
    }
