import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)

try:
    import open_clip
    import torch
except Exception as exc:  # pragma: no cover - environment-dependent import guard
    open_clip = None
    torch = None
    _OPENCLIP_IMPORT_ERROR = exc
else:
    _OPENCLIP_IMPORT_ERROR = None

_model = None
_preprocess = None
_device = None
_tokenizer = None


def _normalize_embedding(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values)
    if norm == 0:
        return values.astype(np.float32)
    return (values / norm).astype(np.float32)


def get_embedding_model():
    global _model, _preprocess, _device
    if open_clip is None or torch is None:
        raise RuntimeError(f'OpenCLIP is unavailable: {_OPENCLIP_IMPORT_ERROR}')

    if _model is None or _preprocess is None:
        _model, _, _preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        _device = 'cpu'
        _model.to(_device)
        _model.eval()
    return _model, _preprocess


def encode_text_embeddings(texts: list[str]) -> np.ndarray:
    global _tokenizer
    model, _ = get_embedding_model()
    if _tokenizer is None:
        _tokenizer = open_clip.get_tokenizer('ViT-B-32')
    tokens = _tokenizer(texts).to(_device)
    with torch.no_grad():
        embeddings = model.encode_text(tokens)
    values = embeddings.cpu().numpy().astype(np.float32)
    return _normalize_embedding_rows(values)


def _normalize_embedding_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return (values / np.maximum(norms, 1e-12)).astype(np.float32)


def compute_embedding_path(record_id: str, project_root: str | None = None) -> str:
    project_root_path = Path(project_root) if project_root else Path(__file__).resolve().parents[3]
    embedding_dir = project_root_path / 'data' / 'embeddings'
    embedding_dir.mkdir(parents=True, exist_ok=True)
    filename = hashlib.sha256(record_id.encode('utf-8')).hexdigest() + '.npy'
    return str(embedding_dir / filename)


def generate_embedding(image_path: str, record_id: str, project_root: str | None = None) -> Dict[str, Any]:
    result = {
        'embedding_path': '',
        'embedding_status': 'failed',
        'error': None,
    }

    try:
        model, preprocess = get_embedding_model()
        from PIL import Image

        image = Image.open(image_path).convert('RGB')
        processed = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            embedding = model.encode_image(processed)
        embedding_array = _normalize_embedding(embedding[0].cpu().numpy())
        embedding_path = compute_embedding_path(record_id, project_root)
        np.save(embedding_path, embedding_array)

        result['embedding_path'] = embedding_path
        result['embedding_status'] = 'completed'
        return result
    except Exception as exc:  # pragma: no cover - fail-safe behavior
        logger.warning('OpenCLIP embedding generation failed for %s: %s', image_path, exc)
        result['error'] = str(exc)
        result['embedding_status'] = 'failed'
        return result
