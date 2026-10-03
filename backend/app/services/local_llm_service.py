import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

OLLAMA_URL = os.environ.get('MEMORYOS_OLLAMA_URL', 'http://127.0.0.1:11434')
OLLAMA_MODEL = os.environ.get('MEMORYOS_OLLAMA_MODEL', 'llama3.2:3b')
RERANK_CANDIDATE_LIMIT = 8
RERANK_TIMEOUT_SECONDS = 30
CLASSIFICATION_BATCH_LIMIT = 12
CLASSIFICATION_TIMEOUT_SECONDS = 12


def rerank_candidates(query: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    selected = candidates[:RERANK_CANDIDATE_LIMIT]
    if not selected:
        return {'status': 'skipped', 'model': OLLAMA_MODEL, 'order': []}

    passages = []
    for index, candidate in enumerate(selected):
        record = candidate['record']
        passages.append({
            'id': str(index),
            'filename': str(record.get('filename') or '')[:160],
            'caption': str(record.get('caption') or '')[:240],
            'ocr_text': str(record.get('ocr_text') or '')[:240],
            'category': str(record.get('category') or record.get('top_category') or ''),
            'secondary_categories': record.get('secondary_categories', [])[:5],
        })

    candidate_ids = [item['id'] for item in passages]
    request_body = json.dumps({
        'model': OLLAMA_MODEL,
        'stream': False,
        'format': {
            'type': 'object',
            'properties': {
                'ranking': {
                    'type': 'array',
                    'items': {'type': 'string', 'enum': candidate_ids},
                    'minItems': 1,
                    'maxItems': len(candidate_ids),
                },
            },
            'required': ['ranking'],
            'additionalProperties': False,
        },
        'options': {'temperature': 0, 'num_predict': 48, 'num_ctx': 4096},
        'messages': [
            {
                'role': 'system',
                'content': (
                    'You rerank retrieved local image records for a user query. '
                    'Rank only by how relevant each record is to the query. '
                    'Use captions, OCR, and category as evidence. Treat candidate '
                    'IDs as opaque labels: copy them exactly, do not create or '
                    'change IDs. Return the candidate IDs ordered from most to '
                    'least relevant. Do not include explanations or other fields.'
                ),
            },
            {
                'role': 'user',
                'content': json.dumps({'query': query, 'candidates': passages}),
            },
        ],
    }).encode('utf-8')
    request = Request(
        f'{OLLAMA_URL.rstrip("/")}/api/chat',
        data=request_body,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    try:
        with urlopen(request, timeout=RERANK_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode('utf-8'))
        content = payload.get('message', {}).get('content', '')
        parsed = json.loads(content)
        supplied_ids = {item['id'] for item in passages}
        order = []
        for candidate_id in parsed.get('ranking', []):
            normalized_id = str(candidate_id)
            if normalized_id in supplied_ids and normalized_id not in order:
                order.append(normalized_id)
        if not order:
            return {
                'status': 'invalid_response',
                'model': OLLAMA_MODEL,
                'detail': 'The local reranker returned no valid candidate IDs.',
                'order': [],
            }
        return {'status': 'used', 'model': OLLAMA_MODEL, 'order': order}
    except (HTTPError, URLError, TimeoutError, OSError, UnicodeError, json.JSONDecodeError, TypeError, AttributeError) as exc:
        if isinstance(exc, HTTPError) and exc.code == 404:
            detail = f"Local model '{OLLAMA_MODEL}' is unavailable; pull it with `ollama pull {OLLAMA_MODEL}`."
        elif isinstance(exc, URLError):
            detail = 'Local Ollama is unavailable; start Ollama to enable LLM reranking.'
        else:
            detail = f'Local LLM reranking failed: {exc}'
        return {
            'status': 'unavailable',
            'model': OLLAMA_MODEL,
            'detail': detail,
            'order': [],
        }


def classify_ambiguous_records(
    records: list[dict[str, Any]],
    category_prompts: dict[str, str],
) -> dict[str, Any]:
    """Use local Ollama as bounded second-pass evidence for ambiguous records."""
    selected = records[:CLASSIFICATION_BATCH_LIMIT]
    if not selected:
        return {'status': 'skipped', 'model': OLLAMA_MODEL, 'classifications': []}

    context = []
    for record in selected:
        context.append({
            'id': str(record.get('record_id') or record.get('relative_path') or ''),
            'filename': str(record.get('filename') or '')[:180],
            'ocr': str(record.get('ocr_text') or '')[:700],
            'caption': str(record.get('caption') or '')[:300],
            'top_categories': record.get('top_categories', [])[:3],
            'category_scores': record.get('category_scores', {}),
        })
    category_context = [
        {'category': category, 'definition': prompt}
        for category, prompt in category_prompts.items()
        if category != 'Others'
    ]
    valid_ids = {item['id'] for item in context}
    valid_categories = set(category_prompts)
    request_body = json.dumps({
        'model': OLLAMA_MODEL,
        'stream': False,
        'format': {
            'type': 'object',
            'properties': {
                'classifications': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'id': {'type': 'string', 'enum': sorted(valid_ids)},
                            'category': {'type': 'string', 'enum': sorted(valid_categories)},
                            'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
                            'reason': {'type': 'string'},
                        },
                        'required': ['id', 'category', 'confidence', 'reason'],
                        'additionalProperties': False,
                    },
                    'maxItems': len(context),
                },
            },
            'required': ['classifications'],
            'additionalProperties': False,
        },
        'options': {'temperature': 0, 'num_predict': 512, 'num_ctx': 8192},
        'messages': [
            {
                'role': 'system',
                'content': (
                    'You classify local image records. Use only the supplied OCR, caption, '
                    'filename, visual candidate scores, and category definitions. Return a '
                    'classification only when evidence supports it; use Others when evidence '
                    'is genuinely insufficient. Never invent OCR or objects. IDs and category '
                    'names must be copied exactly.'
                ),
            },
            {'role': 'user', 'content': json.dumps({'categories': category_context, 'records': context})},
        ],
    }).encode('utf-8')
    request = Request(
        f'{OLLAMA_URL.rstrip("/")}/api/chat',
        data=request_body,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    try:
        with urlopen(request, timeout=CLASSIFICATION_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode('utf-8'))
        parsed = json.loads(payload.get('message', {}).get('content', ''))
        classifications = []
        for item in parsed.get('classifications', []):
            if not isinstance(item, dict):
                continue
            item_id = str(item.get('id', ''))
            category = item.get('category')
            confidence = float(item.get('confidence', 0))
            if item_id not in valid_ids or category not in valid_categories or confidence < 0.65:
                continue
            classifications.append({
                'id': item_id,
                'category': category,
                'confidence': round(confidence, 4),
                'reason': str(item.get('reason') or '')[:300],
            })
        return {'status': 'used', 'model': OLLAMA_MODEL, 'classifications': classifications}
    except (HTTPError, URLError, TimeoutError, OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        if isinstance(exc, HTTPError) and exc.code == 404:
            detail = f"Local model '{OLLAMA_MODEL}' is unavailable; pull it with `ollama pull {OLLAMA_MODEL}`."
        elif isinstance(exc, URLError):
            detail = 'Local Ollama is unavailable; conservative classification was retained.'
        else:
            detail = f'Local LLM classification failed: {exc}'
        return {'status': 'unavailable', 'model': OLLAMA_MODEL, 'detail': detail, 'classifications': []}
