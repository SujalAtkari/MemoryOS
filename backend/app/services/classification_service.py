import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.services.embedding_service import encode_text_embeddings
from app.services.index_storage import save_json_atomically, serialize_index_update
from app.services.library_service import INDEX_PATH, validate_folder_path
from app.services.local_llm_service import classify_ambiguous_records

CLASSIFICATION_VERSION = 'phase3-multi-evidence-v6'
SEMANTIC_TEMPERATURE = 0.08

CATEGORY_PROMPTS = {
    'Government & Identity': 'This image shows a government document or identity document, such as a passport, Aadhaar card, PAN card, or official ID.',
    'Education': 'This image shows education material, such as a marksheet, certificate, school, college, exam, or academic record.',
    'Medical & Health': 'This image shows medical or health information, such as a prescription, medicine, hospital record, or health document.',
    'Finance': 'This image shows finance or banking information, such as a bank statement, account, transaction, payment, or investment record.',
    'Bills & Receipts': 'This image shows a bill, invoice, receipt, purchase record, or payment receipt.',
    'Work & Professional': 'This image shows work or professional material, such as an office, meeting, presentation, business card, or work document.',
    'Travel': 'This image shows travel information, such as a flight, train, ticket, boarding pass, hotel, or travel destination.',
    'Events & Celebrations': 'This image shows an event or celebration, such as a birthday, wedding, party, ceremony, or festival.',
    'People & Family': 'This image shows people, family members, friends, or a personal portrait.',
    'Nature & Places': 'This image shows nature, landscapes, outdoor scenery, buildings, or places.',
    'Animals & Pets': 'This image shows an animal or pet, such as a dog, cat, or other wildlife.',
    'Food & Drinks': 'This image shows food, drinks, a restaurant, meal, recipe, or menu.',
    'Screenshots': 'This image is a computer or phone screenshot showing an application, interface, or captured screen.',
    'Notes & Documents': 'This image shows handwritten notes, a text document, or general paperwork.',
    'Others': 'This image does not clearly fit any of the other image categories.',
}

CATEGORY_KEYWORDS = {
    'Government & Identity': (
        'government', 'aadhaar', 'aadhar', 'pan card', 'passport', 'identity', 'identity card',
        'voter id', 'driving licence', 'driver license', 'ration card', 'uidai',
    ),
    'Education': (
        'education', 'marksheet', 'mark sheet', 'certificate', 'college', 'university',
        'school', 'student', 'exam', 'examination', 'grade', 'transcript', 'degree',
    ),
    'Medical & Health': (
        'medical', 'health', 'prescription', 'medicine', 'medication', 'hospital',
        'doctor', 'patient', 'diagnosis', 'pharmacy', 'clinic', 'laboratory', 'lab report',
    ),
    'Finance': (
        'finance', 'bank', 'banking', 'transaction', 'account number', 'account statement',
        'investment', 'loan', 'credit card', 'debit card', 'transfer', 'upi', 'balance',
    ),
    'Bills & Receipts': (
        'bill', 'receipt', 'invoice', 'purchase', 'tax invoice', 'payment receipt',
        'subtotal', 'total due', 'amount due', 'order total',
    ),
    'Work & Professional': (
        'work', 'professional', 'office', 'meeting', 'business', 'employee',
        'presentation', 'agenda', 'resume', 'curriculum vitae', 'cv', 'contract',
    ),
    'Travel': (
        'travel', 'flight', 'airline', 'boarding pass', 'train', 'railway', 'ticket',
        'hotel', 'booking', 'itinerary', 'departure', 'arrival', 'reservation',
    ),
    'Events & Celebrations': (
        'birthday', 'wedding', 'celebration', 'party', 'ceremony', 'festival',
        'anniversary', 'invitation', 'congratulations', 'event',
    ),
    'People & Family': (
        'family', 'people', 'portrait', 'selfie', 'friends', 'mother', 'father',
        'sister', 'brother', 'children', 'person',
    ),
    'Nature & Places': (
        'nature', 'landscape', 'mountain', 'beach', 'forest', 'lake', 'waterfall',
        'cityscape', 'building', 'park', 'scenery', 'place',
    ),
    'Animals & Pets': (
        'animal', 'pet', 'dog', 'puppy', 'cat', 'kitten', 'bird', 'horse',
        'wildlife', 'fish', 'squirrel', 'monkey', 'primate', 'rabbit', 'bunny',
        'hamster', 'guinea pig', 'turtle', 'tortoise', 'lizard', 'snake',
        'parrot', 'parakeet', 'deer', 'fox', 'bear', 'elephant', 'tiger',
        'lion', 'zebra', 'giraffe', 'koala', 'panda', 'raccoon', 'otter',
    ),
    'Food & Drinks': (
        'food', 'drink', 'restaurant', 'menu', 'meal', 'recipe', 'dish',
        'coffee', 'tea', 'dessert', 'lunch', 'dinner', 'breakfast',
    ),
    'Screenshots': (
        'screenshot', 'screen shot', 'screen capture', 'screencap', 'screenshot',
        'notification', 'status bar', 'app screen',
    ),
    'Notes & Documents': (
        'note', 'notes', 'document', 'paperwork', 'handwritten', 'memo',
        'letter', 'form', 'page', 'text document',
    ),
    'Others': (),
}

CATEGORY_CONCEPTS = {
    'Government & Identity': ('passport', 'identity card', 'driving license', 'visa', 'official identification'),
    'Education': ('marksheet', 'examination paper', 'college document', 'certificate', 'lecture notes'),
    'Medical & Health': ('prescription', 'medical report', 'hospital document', 'medicine', 'health report'),
    'Finance': ('bank statement', 'transaction', 'account statement', 'cheque', 'payment confirmation'),
    'Bills & Receipts': ('shopping receipt', 'restaurant bill', 'invoice', 'electricity bill', 'tax invoice'),
    'Work & Professional': ('office document', 'business presentation', 'work meeting', 'resume', 'business card'),
    'Travel': ('airport', 'airplane', 'boarding pass', 'hotel', 'travel destination'),
    'Events & Celebrations': ('wedding ceremony', 'bride and groom', 'birthday celebration', 'party', 'festival'),
    'People & Family': ('family group', 'portrait', 'friends', 'child', 'people together'),
    'Nature & Places': ('mountain landscape', 'lake', 'beach', 'forest', 'outdoor landscape'),
    'Animals & Pets': (
        'dog', 'cat', 'bird', 'pet', 'wildlife', 'squirrel', 'monkey',
        'rabbit', 'hamster', 'primate', 'wild animal',
    ),
    'Food & Drinks': ('cooked food', 'restaurant meal', 'dessert', 'beverage', 'food plate'),
    'Screenshots': ('mobile screenshot', 'desktop screenshot', 'application interface', 'chat screenshot', 'settings screen'),
    'Notes & Documents': ('handwritten note', 'typed document', 'scanned paper', 'notebook page', 'generic document'),
    'Others': (),
}

ANIMAL_SPECIFIC_KEYWORDS = frozenset({
    'dog', 'puppy', 'cat', 'kitten', 'bird', 'horse', 'fish', 'squirrel',
    'monkey', 'primate', 'rabbit', 'bunny', 'hamster', 'guinea pig',
    'turtle', 'tortoise', 'lizard', 'snake', 'parrot', 'parakeet', 'deer',
    'fox', 'bear', 'elephant', 'tiger', 'lion', 'zebra', 'giraffe', 'koala',
    'panda', 'raccoon', 'otter',
})


@dataclass(frozen=True)
class ClassificationConfig:
    weights: dict[str, float] = field(default_factory=lambda: {
        'filename_score': 0.20,
        'ocr_score': 0.32,
        'caption_score': 0.20,
        'semantic_score': 0.23,
        'metadata_score': 0.05,
    })
    close_category_ratio: float = 0.90
    minimum_ocr_confidence: float = 0.50
    conflict_score: float = 0.70
    minimum_category_score: float = 0.12


DEFAULT_CONFIG = ClassificationConfig()
_category_text_embeddings: np.ndarray | None = None
_concept_text_embeddings: dict[str, np.ndarray] | None = None
_category_embedding_error: str | None = None
_WORD_PATTERN = re.compile(r"[a-z0-9]+")


def _library_id_for_path(folder_path: str) -> str:
    return hashlib.sha256(os.path.normpath(folder_path).encode('utf-8')).hexdigest()[:12]


def _load_index() -> dict:
    if not INDEX_PATH.exists():
        return {'version': 1, 'libraries': {}}
    with INDEX_PATH.open('r', encoding='utf-8') as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get('libraries'), dict):
        raise ValueError('The local library index has an invalid structure.')
    return payload


def _save_index(payload: dict) -> None:
    save_json_atomically(INDEX_PATH, payload)


def _get_category_text_embeddings() -> np.ndarray:
    global _category_text_embeddings, _category_embedding_error
    if _category_text_embeddings is None:
        prompts = [CATEGORY_PROMPTS[category] for category in CATEGORY_PROMPTS]
        try:
            embeddings = np.asarray(encode_text_embeddings(prompts), dtype=np.float32)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            _category_text_embeddings = embeddings / np.maximum(norms, 1e-12)
            _category_embedding_error = None
        except Exception as exc:
            _category_embedding_error = str(exc)
            raise
    return _category_text_embeddings


def _semantic_category_scores(record: dict) -> dict[str, float]:
    embedding_path = record.get('embedding_path')
    if record.get('embedding_status') != 'completed' or not embedding_path:
        return {}

    path = Path(embedding_path)
    if not path.is_file():
        return {}

    vector = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32).reshape(-1)
    if vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError('The image embedding is empty or contains non-finite values.')
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        raise ValueError('The image embedding has zero norm.')
    vector /= norm

    category_vectors = _get_category_text_embeddings()
    if category_vectors.ndim != 2 or category_vectors.shape[1] != vector.size:
        raise ValueError('Image and category text embeddings have incompatible dimensions.')

    similarities = category_vectors @ vector
    logits = (similarities - np.max(similarities)) / SEMANTIC_TEMPERATURE
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()
    return dict(zip(CATEGORY_PROMPTS, probabilities.astype(float), strict=True))


def _semantic_concept_scores(record: dict) -> dict[str, list[dict[str, float]]]:
    global _concept_text_embeddings
    embedding_path = record.get('embedding_path')
    if record.get('embedding_status') != 'completed' or not embedding_path:
        return {}
    path = Path(embedding_path)
    if not path.is_file():
        return {}
    vector = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if vector.size == 0 or not np.isfinite(vector).all() or norm <= 0:
        return {}
    vector /= norm
    if _concept_text_embeddings is None:
        concepts = [concept for values in CATEGORY_CONCEPTS.values() for concept in values]
        encoded = np.asarray(encode_text_embeddings(concepts), dtype=np.float32)
        encoded /= np.maximum(np.linalg.norm(encoded, axis=1, keepdims=True), 1e-12)
        _concept_text_embeddings = {}
        offset = 0
        for category, values in CATEGORY_CONCEPTS.items():
            _concept_text_embeddings[category] = encoded[offset:offset + len(values)]
            offset += len(values)
    result: dict[str, list[dict[str, float]]] = {}
    for category, concepts in CATEGORY_CONCEPTS.items():
        vectors = _concept_text_embeddings.get(category)
        if not concepts or vectors is None or vectors.shape[1] != vector.size:
            continue
        similarities = vectors @ vector
        result[category] = [
            {'concept': concept, 'score': round(float(score), 4)}
            for concept, score in zip(concepts, similarities, strict=True)
            if float(score) >= 0.20
        ]
    return result


def _find_matches(text: str) -> dict[str, list[str]]:
    normalized = ' '.join(_WORD_PATTERN.findall(text.lower()))
    matches: dict[str, list[str]] = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        category_matches = []
        for keyword in keywords:
            normalized_keyword = ' '.join(_WORD_PATTERN.findall(keyword.lower()))
            if normalized_keyword and f' {normalized_keyword} ' in f' {normalized} ':
                category_matches.append(keyword)
        if category_matches:
            matches[category] = sorted(set(category_matches))
    return matches


def _keyword_scores(matches: dict[str, list[str]]) -> dict[str, float]:
    scores = {category: 0.0 for category in CATEGORY_PROMPTS}
    for category, keywords in matches.items():
        if category == 'Animals & Pets' and ANIMAL_SPECIFIC_KEYWORDS.intersection(keywords):
            scores[category] = 1.0
        else:
            scores[category] = min(1.0, len(keywords) / 2.0)
    return scores


def _filename_is_generic(record: dict) -> bool:
    filename = str(record.get('filename') or '').lower()
    stem = Path(filename).stem
    return bool(
        re.fullmatch(r'(?:img|dsc|photo|image)[ _-]?\d{1,}', stem)
        or re.fullmatch(r'screenshot[_ -].*', stem)
        or re.fullmatch(r'[0-9a-f]{8,}', stem)
        or stem.startswith('whatsapp image ')
    )


def _concept_matches(text: str) -> dict[str, list[str]]:
    normalized = ' '.join(_WORD_PATTERN.findall(text.lower()))
    matches: dict[str, list[str]] = {}
    for category, concepts in CATEGORY_CONCEPTS.items():
        found = [
            concept for concept in concepts
            if f" {' '.join(_WORD_PATTERN.findall(concept.lower()))} " in f" {normalized} "
        ]
        if found:
            matches[category] = found
    return matches


def _normalize_available_weights(available_sources: list[str], config: ClassificationConfig) -> dict[str, float]:
    total = sum(config.weights[source] for source in available_sources)
    if total <= 0:
        return {}
    return {source: config.weights[source] / total for source in available_sources}


def classify_record(
    record: dict,
    semantic_scores: dict[str, float] | None = None,
    semantic_concepts: dict[str, list[dict[str, float]]] | None = None,
    config: ClassificationConfig = DEFAULT_CONFIG,
) -> dict:
    filename_matches = _find_matches(f"{record.get('filename', '')} {record.get('relative_path', '')}")
    caption_matches = _find_matches(str(record.get('caption') or ''))
    ocr_text = str(record.get('ocr_text') or '')
    ocr_confidence = record.get('ocr_confidence')
    ocr_reliable = (
        record.get('ocr_status') == 'completed'
        and isinstance(ocr_confidence, (int, float))
        and float(ocr_confidence) >= config.minimum_ocr_confidence
    )
    ocr_matches = _find_matches(ocr_text) if ocr_reliable else {}

    source_scores: dict[str, dict[str, float]] = {}
    if filename_matches:
        filename_scores = _keyword_scores(filename_matches)
        if _filename_is_generic(record):
            filename_scores = {category: score * 0.15 for category, score in filename_scores.items()}
        source_scores['filename_score'] = filename_scores
    if ocr_matches:
        source_scores['ocr_score'] = _keyword_scores(ocr_matches)
    if caption_matches:
        source_scores['caption_score'] = _keyword_scores(caption_matches)
    if semantic_scores:
        source_scores['semantic_score'] = {
            category: max(0.0, float(semantic_scores.get(category, 0.0)))
            for category in CATEGORY_PROMPTS
        }

    adaptive_config_weights = dict(config.weights)
    if ocr_reliable and 'ocr_score' in source_scores:
        adaptive_config_weights['ocr_score'] *= 1.25
    if semantic_scores and 'semantic_score' in source_scores:
        adaptive_config_weights['semantic_score'] *= 1.15
    weights = _normalize_available_weights(
        list(source_scores),
        ClassificationConfig(
            weights=adaptive_config_weights,
            close_category_ratio=config.close_category_ratio,
            minimum_ocr_confidence=config.minimum_ocr_confidence,
            conflict_score=config.conflict_score,
            minimum_category_score=config.minimum_category_score,
        ),
    )
    category_scores = {
        category: sum(
            weights[source] * source_scores[source].get(category, 0.0)
            for source in weights
        )
        for category in CATEGORY_PROMPTS
    }
    if semantic_concepts:
        # Concept prompts are more specific than broad category prompts. Add
        # their strongest visual match so clear objects such as food or drinks
        # can outweigh a generic scene-level category.
        for category, matches in semantic_concepts.items():
            strongest_match = max(
                (float(item.get('score', 0.0)) for item in matches),
                default=0.0,
            )
            category_scores[category] += max(0.0, strongest_match) * 0.20
    ordered = sorted(category_scores.items(), key=lambda item: (-item[1], item[0]))
    top_category, top_score = ordered[0]
    alternative_category, alternative_score = ordered[1]
    evidence_source_winners = {
        source: max(scores.items(), key=lambda item: (item[1], item[0]))
        for source, scores in source_scores.items()
        if max(scores.values(), default=0.0) >= config.conflict_score
    }
    strong_winners = {
        category for category, score in evidence_source_winners.values()
        if score >= config.conflict_score
    }
    conflict = len(strong_winners) > 1
    insufficient_evidence = not source_scores or max(category_scores.values(), default=0.0) == 0
    close_scores = top_score <= 0 or alternative_score / top_score >= config.close_category_ratio
    weak_evidence = top_score < config.minimum_category_score
    needs_fallback = insufficient_evidence or conflict or close_scores or weak_evidence
    status = 'classified'
    category = 'Others' if needs_fallback else top_category

    matched_keywords = {
        'filename': filename_matches,
        'ocr': ocr_matches,
        'caption': caption_matches,
    }
    concept_matches = _concept_matches(
        f"{record.get('filename', '')} {record.get('relative_path', '')} "
        f"{ocr_text} {record.get('caption') or ''}"
    )
    if semantic_concepts:
        concept_matches = {
            category: [
                item['concept'] for item in sorted(
                    items,
                    key=lambda item: item['score'],
                    reverse=True,
                )[:3]
            ]
            for category, items in semantic_concepts.items()
            if items
        } | concept_matches
    secondary_categories = [
        category for category, score in ordered[1:]
        if score >= 0.25 and top_score - score >= 0.02
    ][:2]
    reasons = []
    for source, source_matches in (
        ('filename/path', filename_matches),
        ('OCR text', ocr_matches),
        ('BLIP caption', caption_matches),
    ):
        for matched_category, keywords in source_matches.items():
            reasons.append(f"{source} matched {', '.join(repr(keyword) for keyword in keywords)} for {matched_category}")
    if semantic_scores:
        semantic_winner = max(semantic_scores.items(), key=lambda item: item[1])
        reasons.append(
            f"OpenCLIP category-text similarity favored {semantic_winner[0]} "
            f"({semantic_winner[1]:.3f})"
        )
    if semantic_concepts:
        concept_winner = max(
            (
                (category, item['concept'], item['score'])
                for category, items in semantic_concepts.items()
                for item in items
            ),
            key=lambda item: item[2],
            default=None,
        )
        if concept_winner:
            reasons.append(
                f"OpenCLIP visual concept evidence matched "
                f"{concept_winner[1]} ({concept_winner[2]:.3f})"
            )
    if conflict:
        reasons.append('Strong evidence sources support different categories')
    if insufficient_evidence:
        reasons.append('Insufficient classification evidence')
    if weak_evidence and not insufficient_evidence:
        reasons.append(
            f"Leading category score is below the {config.minimum_category_score:.2f} "
            'minimum; assigned to Others'
        )
    if close_scores and not insufficient_evidence:
        reasons.append(
            f"Runner-up category score is within {config.close_category_ratio:.0%} "
            'of the leading category'
        )
    if needs_fallback:
        reasons.append('Assigned to Others because the available evidence is ambiguous or insufficient')
    if secondary_categories:
        reasons.append(f"Secondary evidence also supports {', '.join(secondary_categories)}")

    confidence = round(float(top_score), 4)
    assignment_reason = (
        'Insufficient evidence; assigned to Others.'
        if insufficient_evidence
        else f'Weak evidence (score below {config.minimum_category_score:.2f}); assigned to Others.'
        if weak_evidence
        else 'Evidence sources conflict; assigned to Others.'
        if conflict
        else 'Leading categories are too close; assigned to Others.'
        if close_scores
        else f'Assigned to the strongest supported category: {top_category}.'
    )
    return {
        'category': category,
        'confidence': confidence,
        'classification_confidence': confidence,
        'top_category': top_category,
        'top_score': round(float(top_score), 4),
        'alternative_category': alternative_category,
        'alternative_score': round(float(alternative_score), 4),
        'secondary_categories': secondary_categories,
        'classification_status': status,
        'classification_version': CLASSIFICATION_VERSION,
        'classification_sha256': record.get('sha256'),
        'classification_reason': reasons,
        'assignment_reason': assignment_reason,
        'review_reasons': [],
        'conflicting_sources': sorted(
            source for source, winner in evidence_source_winners.items()
            if winner[0] != top_category
        ),
        'top_categories': [
            {'category': name, 'score': round(float(score), 4)}
            for name, score in ordered[:3]
        ],
        'matched_concepts': concept_matches,
        'matched_keywords': matched_keywords,
        'evidence_sources': sorted(source_scores),
        'category_scores': {
            name: round(float(score), 4) for name, score in category_scores.items()
        },
        'top_evidence': {
            source: {
                'category': winner[0],
                'score': round(float(winner[1]), 4),
            }
            for source, winner in evidence_source_winners.items()
        },
        'classification_error': None,
    }


def _outputs_are_valid(record: dict) -> bool:
    return (
        record.get('classification_version') == CLASSIFICATION_VERSION
        and record.get('classification_sha256') == record.get('sha256')
        and record.get('classification_status') in {'classified', 'needs_review'}
        and isinstance(record.get('category_scores'), dict)
        and isinstance(record.get('classification_reason'), list)
    )


def _classification_response(relative_path: str, record: dict) -> dict:
    fields = (
        'category',
        'classification_status',
        'classification_confidence',
        'classification_reason',
        'matched_keywords',
        'evidence_sources',
        'top_evidence',
        'category_scores',
        'top_category',
        'top_score',
        'alternative_category',
        'alternative_score',
        'secondary_categories',
        'review_reasons',
        'conflicting_sources',
        'top_categories',
        'matched_concepts',
        'classification_error',
        'assignment_reason',
        'ocr_text',
        'ocr_confidence',
        'ocr_status',
        'caption',
        'caption_status',
        'ai_processing_status',
        'ai_processing_profile',
        'embedding_status',
    )
    return {
        'relative_path': relative_path,
        **{field: record.get(field) for field in fields},
    }


@serialize_index_update
def process_library_classification(
    folder_path: str,
    force: bool = False,
    record_id: str | None = None,
    use_llm: bool = False,
    config: ClassificationConfig = DEFAULT_CONFIG,
) -> dict:
    library_root = validate_folder_path(folder_path)
    library_id = _library_id_for_path(library_root)
    index = _load_index()
    library_index = index.get('libraries', {}).get(library_id)
    if library_index is None:
        raise ValueError('Library has not been scanned yet.')

    records = library_index.get('records', {})
    if record_id is not None and record_id not in records:
        raise ValueError(f'Image record not found: {record_id}')

    total = classified = needs_review = failed = skipped = 0
    errors = []
    classifications = []
    selected_records = (
        [(record_id, records[record_id])]
        if record_id is not None
        else list(records.items())
    )

    for relative_path, record in selected_records:
        if record.get('file_status') == 'missing':
            skipped += 1
            continue
        total += 1
        if (
            not force
            and not use_llm
            and _outputs_are_valid(record)
            and record.get('classification_status') == 'classified'
        ):
            skipped += 1
            classified += 1
            classifications.append(_classification_response(relative_path, record))
            continue

        try:
            semantic_scores = _semantic_category_scores(record)
            semantic_concepts = _semantic_concept_scores(record) if semantic_scores else {}
            result = classify_record(record, semantic_scores, semantic_concepts, config)
            record.update(result)
            classifications.append(_classification_response(relative_path, record))
            classified += 1
        except Exception as exc:
            record.update({
                'category': None,
                'confidence': None,
                'classification_confidence': None,
                'top_category': None,
                'top_score': None,
                'alternative_category': None,
                'alternative_score': None,
                'secondary_categories': [],
                'review_reasons': ['processing_failed'],
                'conflicting_sources': [],
                'top_categories': [],
                'matched_concepts': {},
                'classification_status': 'failed',
                'classification_version': CLASSIFICATION_VERSION,
                'classification_sha256': record.get('sha256'),
                'classification_error': str(exc),
                'classification_reason': [],
                'matched_keywords': {},
                'evidence_sources': [],
                'category_scores': {},
                'top_evidence': {},
            })
            failed += 1
            errors.append({'relative_path': relative_path, 'error': str(exc)})
            classifications.append(_classification_response(relative_path, record))
        records[relative_path] = record

    _save_index(index)
    llm_result = {'status': 'skipped', 'model': None, 'classifications': []}
    if use_llm and not record_id:
        ambiguous_records = [
            record for record in records.values()
            if record.get('category') == 'Others'
            and record.get('classification_status') == 'classified'
        ]
        for offset in range(0, len(ambiguous_records), 12):
            batch_result = classify_ambiguous_records(
                ambiguous_records[offset:offset + 12],
                CATEGORY_PROMPTS,
            )
            if batch_result.get('status') != 'used':
                llm_result = batch_result
                break
            llm_result = batch_result
            accepted = {item['id']: item for item in batch_result.get('classifications', [])}
            for relative_path, record in records.items():
                item = accepted.get(str(record.get('record_id') or relative_path))
                if not item or item['category'] == 'Others':
                    continue
                record['category'] = item['category']
                record['classification_confidence'] = item['confidence']
                record['confidence'] = item['confidence']
                record['classification_reason'] = [
                    *(record.get('classification_reason') or []),
                    f"Local RAG classifier: {item['reason']}",
                ]
                record['assignment_reason'] = f"Assigned by local RAG classifier: {item['reason']}"
                record['classification_sources'] = sorted({
                    *(record.get('evidence_sources') or []),
                    'local_llm_rag',
                })
                records[relative_path] = record
        _save_index(index)
        classifications = [
            _classification_response(relative_path, record)
            for relative_path, record in records.items()
            if record.get('classification_status') in {'classified', 'needs_review'}
        ]
    return {
        'library_id': library_id,
        'library_root': library_root,
        'total': total,
        'classified': classified,
        'needs_review': needs_review,
        'failed': failed,
        'skipped': skipped,
        'processing_status': 'partial_failure' if failed else 'completed',
        'errors': errors,
        'classifications': classifications,
        'classification_version': CLASSIFICATION_VERSION,
        'llm': {
            'status': llm_result.get('status'),
            'model': llm_result.get('model'),
            'detail': llm_result.get('detail'),
            'accepted': len(llm_result.get('classifications', [])),
        },
    }


@serialize_index_update
def approve_classification(
    folder_path: str,
    record_id: str,
    category: str,
) -> dict:
    library_root = validate_folder_path(folder_path)
    if category not in CATEGORY_PROMPTS:
        raise ValueError('Invalid classification category.')

    library_id = _library_id_for_path(library_root)
    index = _load_index()
    library_index = index.get('libraries', {}).get(library_id)
    if library_index is None:
        raise ValueError('Library has not been scanned yet.')

    records = library_index.get('records', {})
    record = records.get(record_id)
    if record is None:
        raise ValueError(f'Image record not found: {record_id}')
    if record.get('file_status') == 'missing':
        raise ValueError('Cannot approve a missing image.')

    record.update({
        'category': category,
        'classification_status': 'classified',
        'classification_error': None,
        'review_reasons': [],
        'classification_reason': [
            *(record.get('classification_reason') or []),
            f'Category approved by the user: {category}',
        ],
        'classification_sha256': record.get('sha256'),
        'classification_version': CLASSIFICATION_VERSION,
    })
    records[record_id] = record
    _save_index(index)
    return _classification_response(record_id, record)
