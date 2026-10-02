import hashlib
from pathlib import Path

from PIL import Image
from imagehash import phash

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tiff', '.tif'}


def is_supported_image(file_path: str) -> bool:
    return Path(file_path).suffix.lower() in SUPPORTED_EXTENSIONS


def compute_sha256(file_path: str) -> str:
    digest = hashlib.sha256()
    with open(file_path, 'rb') as file_handle:
        for chunk in iter(lambda: file_handle.read(65536), b''):
            digest.update(chunk)
    return digest.hexdigest()


def compute_phash_value(file_path: str) -> str:
    with Image.open(file_path) as image:
        rgb_image = image.convert('RGB')
        return str(phash(rgb_image))
