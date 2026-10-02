import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

try:
    import torch
    from paddleocr import PaddleOCR
except Exception as exc:  # pragma: no cover - environment-dependent import guard
    torch = None
    PaddleOCR = None
    _PADDLEOCR_IMPORT_ERROR = exc
else:
    _PADDLEOCR_IMPORT_ERROR = None

_ocr_model = None


def get_ocr_model():
    global _ocr_model
    if PaddleOCR is None:
        raise RuntimeError(f'PaddleOCR is unavailable: {_PADDLEOCR_IMPORT_ERROR}')
    if _ocr_model is None:
        _ocr_model = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    return _ocr_model


def run_ocr(image_path: str) -> Dict[str, Any]:
    result = {
        'text': '',
        'confidence': None,
        'status': 'failed',
        'error': None,
    }

    try:
        ocr_engine = get_ocr_model()
        extracted = ocr_engine.ocr(image_path, cls=True)

        text_lines = []
        confidences = []
        for page in extracted or []:
            for item in page or []:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                info = item[1]
                if isinstance(info, dict):
                    text = (info.get('text') or '').strip()
                    confidence = info.get('confidence')
                    if text:
                        text_lines.append(text)
                    if confidence is not None:
                        confidences.append(float(confidence))
                elif isinstance(info, (list, tuple)) and len(info) >= 2:
                    text = str(info[0]).strip()
                    if text:
                        text_lines.append(text)
                    if info[1] is not None:
                        confidences.append(float(info[1]))
                else:
                    text = str(info).strip()
                    if text:
                        text_lines.append(text)

        result['text'] = '\n'.join(text_lines).strip()
        if confidences:
            result['confidence'] = round(sum(confidences) / len(confidences), 4)
        result['status'] = 'completed'
        return result
    except Exception as exc:  # pragma: no cover - allows safe pipeline fallback
        logger.warning('OCR failed for %s: %s', image_path, exc)
        result['error'] = str(exc)
        result['status'] = 'failed'
        return result
