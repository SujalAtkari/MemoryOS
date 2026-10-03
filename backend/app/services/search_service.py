import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from app.services.embedding_service import encode_text_embeddings
from app.services.library_service import INDEX_PATH, validate_folder_path
from app.services.local_llm_service import rerank_candidates

SEARCH_VERSION = 'phase4-rag-v1'
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'by', 'for', 'from', 'in', 'images',
    'is', 'of', 'on', 'or', 'photo', 'photos', 'picture', 'pictures', 'the',
    'this', 'to', 'with',
}
_QUERY_EXPANSIONS = {
    'bat': ('cricket', 'cricket bat', 'baseball bat', 'sports'),
    'ball': ('cricket', 'cricket ball', 'baseball', 'sports'),
    'cricket': ('bat', 'ball', 'sports'),
    'wedding': ('bride', 'groom', 'ceremony', 'celebration'),
    'bride': ('wedding', 'ceremony', 'celebration'),
    'groom': ('wedding', 'ceremony', 'celebration'),
    'mountain': ('landscape', 'nature', 'scenery'),
    'lake': ('landscape', 'nature', 'scenery'),
    'receipt': ('invoice', 'bill', 'payment'),
    'invoice': ('receipt', 'bill', 'payment'),
}


@dataclass(frozen=True)
class SearchConfig:
    semantic_score: float = 0.50
    ocr_score: float = 0.20
    filename_score: float = 0.10
    caption_score: float = 0.10
    category_score: float = 0.05
    metadata_score: float = 0.05
    minimum_semantic_cosine: float = 0.30

    def weights(self) -> dict[str, float]:
        return {
            'semantic_score': self.semantic_score,
            'ocr_score': self.ocr_score,
            'filename_score': self.filename_score,
            'caption_score': self.caption_score,
            'category_score': self.category_score,
            'metadata_score': self.metadata_score,
        }


DEFAULT_SEARCH_CONFIG = SearchConfig()


@lru_cache(maxsize=5000)
def _load_embedding(path: str, modified_ns: int, size: int) -> np.ndarray:
    del modified_ns, size
    return np.asarray(np.load(path, allow_pickle=False), dtype=np.float32).reshape(-1)


def _tokens(value: Any) -> list[str]:
    return [
        token for token in _TOKEN_PATTERN.findall(str(value or '').lower())
        if token not in _STOP_WORDS
    ]


def _token_score(query_tokens: list[str], text: str) -> tuple[float, list[str]]:
    if not query_tokens:
        return 0.0, []
    content_tokens = set(_tokens(text))
    matched = sorted(set(query_tokens) & content_tokens)
    return len(matched) / len(set(query_tokens)), matched


def _expanded_query_tokens(query_tokens: list[str]) -> list[str]:
    expanded = list(query_tokens)
    for token in query_tokens:
        expanded.extend(_tokens(' '.join(_QUERY_EXPANSIONS.get(token, ()))))
    return sorted(set(expanded))


def _load_index() -> dict:
    if not INDEX_PATH.exists():
        return {'version': 1, 'libraries': {}}
    with INDEX_PATH.open('r', encoding='utf-8') as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get('libraries'), dict):
        raise ValueError('The local library index has an invalid structure.')
    return payload


def _normalize_weights(sources: list[str], weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights[source] for source in sources)
    if total <= 0:
        return {}
    return {source: weights[source] / total for source in sources}


def _record_image_path(library_root: str, record: dict) -> Path:
    relative_path = Path(str(record.get('relative_path') or ''))
    if relative_path.is_absolute():
        return relative_path
    return Path(library_root) / relative_path


def _record_embedding(record: dict) -> np.ndarray | None:
    embedding_path = record.get('embedding_path')
    if record.get('embedding_status') != 'completed' or not embedding_path:
        return None
    try:
        path = Path(embedding_path)
        stat = path.stat()
        vector = _load_embedding(str(path), stat.st_mtime_ns, stat.st_size)
    except (OSError, ValueError, TypeError):
        return None
    if vector.size == 0 or not np.isfinite(vector).all():
        return None
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        return None
    return vector / norm


def _category_text(record: dict) -> str:
    values = [
        record.get('category'),
        record.get('subcategory'),
        *record.get('secondary_categories', []),
        *record.get('keywords', []),
        *record.get('phrases', []),
    ]
    matched_keywords = record.get('matched_keywords')
    if isinstance(matched_keywords, dict):
        for entries in matched_keywords.values():
            if isinstance(entries, dict):
                for keywords in entries.values():
                    if isinstance(keywords, list):
                        values.extend(keywords)
            elif isinstance(entries, list):
                values.extend(entries)
    matched_concepts = record.get('matched_concepts')
    if isinstance(matched_concepts, dict):
        for concepts in matched_concepts.values():
            if isinstance(concepts, list):
                values.extend(concepts)
    return ' '.join(str(value) for value in values if value)


def _search_reason(matched_fields: list[str], matched_keywords: list[str], cosine: float | None) -> str:
    reasons = []
    if matched_fields:
        reasons.append(f"Query terms matched {', '.join(matched_fields)}")
    if matched_keywords:
        reasons.append(f"Matched terms: {', '.join(matched_keywords)}")
    if cosine is not None:
        reasons.append(f"OpenCLIP cosine similarity: {cosine:.3f}")
    return '; '.join(reasons) or 'Ranked by available hybrid evidence'


def search_images(
    query: str,
    folder_path: str | None = None,
    category: str | None = None,
    important_only: bool = False,
    limit: int = 20,
    min_score: float = 0.0,
    use_llm: bool = True,
    config: SearchConfig = DEFAULT_SEARCH_CONFIG,
) -> dict:
    if not isinstance(query, str) or not query.strip():
        raise ValueError('Search query is required.')
    if limit < 1 or limit > 500:
        raise ValueError('Search limit must be between 1 and 500.')
    if not 0 <= min_score <= 1:
        raise ValueError('Minimum score must be between 0 and 1.')

    normalized_root = validate_folder_path(folder_path) if folder_path else None
    index = _load_index()
    query_vector = np.asarray(encode_text_embeddings([query.strip()]), dtype=np.float32).reshape(-1)
    query_norm = float(np.linalg.norm(query_vector))
    if query_vector.size == 0 or not np.isfinite(query_vector).all() or query_norm <= 0:
        raise RuntimeError('OpenCLIP returned an invalid query embedding.')
    query_vector /= query_norm
    query_tokens = _tokens(query)
    expanded_query_tokens = _expanded_query_tokens(query_tokens)
    if 'important' in query_tokens:
        important_only = True
    configured_weights = config.weights()
    allowed_category = category.strip().casefold() if category and category.strip() else None

    candidates: list[dict[str, Any]] = []
    for library_id, library in index['libraries'].items():
        library_root = str(library.get('library_root') or '')
        if normalized_root and Path(library_root).resolve() != Path(normalized_root).resolve():
            continue
        records = library.get('records', {})
        if not isinstance(records, dict):
            continue
        for relative_path, record in records.items():
            if not isinstance(record, dict) or record.get('file_status') in {
                'missing', 'modified', 'unverified',
            }:
                continue
            if record.get('ai_processing_status') == 'failed':
                continue
            if important_only and record.get('is_important') is not True:
                continue
            image_path = _record_image_path(library_root, record)
            if not image_path.is_file():
                continue
            record_category = str(record.get('category') or '')
            suggested_category = str(record.get('top_category') or '')
            if allowed_category and allowed_category not in {
                record_category.casefold(),
                suggested_category.casefold(),
            }:
                continue

            image_vector = _record_embedding(record)
            if image_vector is not None and image_vector.size != query_vector.size:
                image_vector = None

            filename_text = f"{record.get('filename', '')} {relative_path}"
            ocr_text = record.get('ocr_text') or ''
            caption_text = record.get('caption') or ''
            category_text = _category_text(record)
            metadata_text = ' '.join(
                str(record.get(key) or '')
                for key in ('extension', 'width', 'height', 'aspect_ratio')
            )

            field_scores: dict[str, float] = {}
            field_matches: dict[str, list[str]] = {}
            for field_name, text in (
                ('filename', filename_text),
                ('ocr', ocr_text),
                ('caption', caption_text),
                ('category', category_text),
                ('metadata', metadata_text),
            ):
                if text:
                    field_scores[field_name], field_matches[field_name] = _token_score(
                        expanded_query_tokens,
                        str(text),
                    )

            matched_fields = [name for name, score in field_scores.items() if score > 0]
            matched_keywords = sorted({
                term for terms in field_matches.values() for term in terms
            })
            if allowed_category and record_category.casefold() == allowed_category:
                matched_fields.append('category_filter')

            sources = ['semantic_score'] if image_vector is not None else []
            source_values: dict[str, float] = (
                {'semantic_score': 0.0} if image_vector is not None else {}
            )
            source_map = {
                'filename': 'filename_score',
                'ocr': 'ocr_score',
                'caption': 'caption_score',
                'category': 'category_score',
                'metadata': 'metadata_score',
            }
            for field_name, source_name in source_map.items():
                if field_name in field_scores:
                    sources.append(source_name)
                    source_values[source_name] = field_scores[field_name]

            candidates.append({
                'library_id': library_id,
                'library_root': library_root,
                'relative_path': relative_path,
                'record': record,
                'image_vector': image_vector,
                'field_scores': field_scores,
                'sources': sources,
                'source_values': source_values,
                'matched_fields': sorted(set(matched_fields)),
                'matched_keywords': matched_keywords,
                'query_expansions': sorted(set(expanded_query_tokens) - set(query_tokens)),
            })

    if not candidates:
        return {
            'query': query,
            'category': category,
            'total_results': 0,
            'returned_results': 0,
            'reranker': {
                'status': 'skipped' if use_llm else 'disabled',
                'model': None,
                'detail': None,
            },
            'results': [],
        }

    embedded_candidates = [
        candidate for candidate in candidates if candidate['image_vector'] is not None
    ]
    if embedded_candidates:
        image_matrix = np.stack([candidate['image_vector'] for candidate in embedded_candidates])
        cosine_values = image_matrix @ query_vector
        cosine_by_candidate = {
            id(candidate): float(np.clip(cosine, -1.0, 1.0))
            for candidate, cosine in zip(embedded_candidates, cosine_values, strict=True)
        }
    else:
        cosine_by_candidate = {}
    for candidate in candidates:
        semantic_cosine = cosine_by_candidate.get(id(candidate))
        candidate['semantic_cosine'] = semantic_cosine
        if semantic_cosine is not None:
            candidate['source_values']['semantic_score'] = (semantic_cosine + 1.0) / 2.0
        normalized = _normalize_weights(candidate['sources'], configured_weights)
        candidate['hybrid_score'] = sum(
            normalized[source] * candidate['source_values'][source]
            for source in normalized
        )
        lexical_sources = [source for source in candidate['sources'] if source != 'semantic_score']
        lexical_weights = _normalize_weights(lexical_sources, configured_weights)
        candidate['lexical_score'] = (
            sum(
                lexical_weights[source] * candidate['source_values'][source]
                for source in lexical_weights
            )
            if lexical_weights
            else 0.0
        )

    filtered_candidates = [
        candidate for candidate in candidates
        if candidate['lexical_score'] > 0
        or (
            candidate['semantic_cosine'] is not None
            and candidate['semantic_cosine'] >= config.minimum_semantic_cosine
        )
    ]
    ranked = sorted(
        filtered_candidates,
        key=lambda candidate: (
            -candidate['hybrid_score'],
            -candidate['lexical_score'],
            -(candidate['semantic_cosine'] if candidate['semantic_cosine'] is not None else -1.0),
            candidate['relative_path'].casefold(),
        ),
    )
    filtered = [candidate for candidate in ranked if candidate['hybrid_score'] >= min_score]
    rerank_result = (
        rerank_candidates(query.strip(), filtered)
        if use_llm and filtered
        else {'status': 'disabled' if not use_llm else 'skipped', 'model': None, 'order': []}
    )
    reranked_candidate_ids: set[int] = set()
    if rerank_result['status'] == 'used':
        candidate_by_id = {
            str(index): candidate
            for index, candidate in enumerate(filtered[:30])
        }
        reranked_candidate_ids = {
            id(candidate_by_id[candidate_id])
            for candidate_id in rerank_result['order']
            if candidate_id in candidate_by_id
        }
        reranked = [
            candidate_by_id[candidate_id]
            for candidate_id in rerank_result['order']
            if candidate_id in candidate_by_id
        ]
        seen_ids = {id(candidate) for candidate in reranked}
        filtered = reranked + [
            candidate for candidate in filtered
            if id(candidate) not in seen_ids
        ]
    results = []
    for rank, candidate in enumerate(filtered[:limit], start=1):
        record = candidate['record']
        cosine = candidate['semantic_cosine']
        results.append({
            'record_id': record.get('record_id', candidate['relative_path']),
            'filename': record.get('filename', Path(candidate['relative_path']).name),
            'relative_path': candidate['relative_path'],
            'category': record.get('category'),
            'subcategory': record.get('subcategory'),
            'is_important': record.get('is_important') is True,
            'keywords': record.get('keywords', []),
            'phrases': record.get('phrases', []),
            'classification_confidence': record.get('classification_confidence'),
            'semantic_score': round(cosine, 6) if cosine is not None else None,
            'lexical_score': round(candidate['lexical_score'], 6),
            'hybrid_score': round(candidate['hybrid_score'], 6),
            'relevance_rank': rank,
            'reranked_by_llm': id(candidate) in reranked_candidate_ids,
            'matched_fields': candidate['matched_fields'],
            'matched_keywords': candidate['matched_keywords'],
            'query_expansions': candidate['query_expansions'],
            'search_reason': _search_reason(
                candidate['matched_fields'],
                candidate['matched_keywords'],
                cosine,
            ),
            'classification_status': record.get('classification_status'),
            'classification_reason': record.get('classification_reason', []),
            'matched_classification_keywords': record.get('matched_keywords', {}),
            'evidence_sources': record.get('evidence_sources', []),
            'top_evidence': record.get('top_evidence', {}),
            'category_scores': record.get('category_scores', {}),
            'top_category': record.get('top_category'),
            'top_score': record.get('top_score'),
            'alternative_category': record.get('alternative_category'),
            'alternative_score': record.get('alternative_score'),
            'secondary_categories': record.get('secondary_categories', []),
            'review_reasons': record.get('review_reasons', []),
            'conflicting_sources': record.get('conflicting_sources', []),
            'top_categories': record.get('top_categories', []),
            'matched_concepts': record.get('matched_concepts', {}),
            'ocr_text': record.get('ocr_text'),
            'ocr_confidence': record.get('ocr_confidence'),
            'ocr_status': record.get('ocr_status'),
            'caption': record.get('caption'),
            'caption_status': record.get('caption_status'),
            'ai_processing_profile': record.get('ai_processing_profile'),
            'embedding_status': record.get('embedding_status'),
            'file_status': record.get('file_status'),
            'processing_status': record.get('processing_status'),
            'ai_processing_status': record.get('ai_processing_status'),
            'library_id': candidate['library_id'],
            'library_root': candidate['library_root'],
        })

    return {
        'query': query,
        'category': category,
        'total_results': len(filtered),
        'returned_results': len(results),
        'reranker': {
            'status': rerank_result['status'],
            'model': rerank_result.get('model'),
            'detail': rerank_result.get('detail'),
        },
        'results': results,
    }
