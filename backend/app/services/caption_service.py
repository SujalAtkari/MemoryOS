import logging
from typing import Any, Dict

from PIL import Image

logger = logging.getLogger(__name__)

try:
    from transformers import pipeline
    import torch
except Exception as exc:  # pragma: no cover - environment-dependent import guard
    pipeline = None
    torch = None
    _BLIP_IMPORT_ERROR = exc
else:
    _BLIP_IMPORT_ERROR = None

_caption_pipeline = None


def get_caption_pipeline():
    global _caption_pipeline
    if pipeline is None:
        raise RuntimeError(f'BLIP dependencies are unavailable: {_BLIP_IMPORT_ERROR}')
    if _caption_pipeline is None:
        _caption_pipeline = pipeline(
            'image-to-text',
            model='Salesforce/blip-image-captioning-base',
            device=-1,
        )
    return _caption_pipeline


def generate_caption(image_path: str) -> Dict[str, Any]:
    result = {
        'caption': '',
        'status': 'failed',
        'error': None,
    }

    try:
        captioner = get_caption_pipeline()
        with Image.open(image_path) as source:
            image = source.convert('RGB')
        if image.width < 32 or image.height < 32:
            scale = max(32 / image.width, 32 / image.height)
            size = (round(image.width * scale), round(image.height * scale))
            image = image.resize(size)
        output = captioner(image, max_new_tokens=24)
        caption = output[0].get('generated_text') if isinstance(output, list) and output else ''
        result['caption'] = str(caption).strip() if caption else ''
        result['status'] = 'completed' if result['caption'] else 'failed'
        if not result['caption']:
            result['error'] = 'The caption model returned no text for this image.'
        return result
    except Exception as exc:  # pragma: no cover - safe failure behavior
        logger.warning('BLIP caption generation failed for %s: %s', image_path, exc)
        result['error'] = str(exc)
        result['status'] = 'failed'
        return result
