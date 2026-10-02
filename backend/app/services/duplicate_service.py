import hashlib
import json
import os
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np

from app.services.library_service import INDEX_PATH, validate_folder_path
from app.services.index_storage import save_json_atomically, serialize_index_update
from app.services.search_service import _record_embedding

DUPLICATE_VERSION = 'phase5-duplicates-v1'
PHASH_HAMMING_THRESHOLD = 8
OPENCLIP_DUPLICATE_THRESHOLD = 0.90
VISUAL_CANDIDATE_PHASH_DISTANCE = 16


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


def _record_image_path(library_root: str, record: dict) -> Path:
    relative_path = Path(str(record.get('relative_path') or ''))
    if relative_path.is_absolute():
        return relative_path
    return Path(library_root) / relative_path


def _valid_sha256(value: object) -> str | None:
    if not isinstance(value, str) or len(value) != 64:
        return None
    try:
        int(value, 16)
    except ValueError:
        return None
    return value.lower()


def _valid_phash(value: object) -> tuple[int, int] | None:
    if not isinstance(value, str) or not value or len(value) > 256:
        return None
    try:
        bits = len(value) * 4
        return int(value, 16), bits
    except ValueError:
        return None


def _compute_sha256(image_path: Path) -> str:
    digest = hashlib.sha256()
    with image_path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(65536), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _phash_distance(left: str, right: str) -> int | None:
    left_value = _valid_phash(left)
    right_value = _valid_phash(right)
    if left_value is None or right_value is None or left_value[1] != right_value[1]:
        return None
    return (left_value[0] ^ right_value[0]).bit_count()


def _candidate_phash_pairs(phashes: dict[str, str], maximum_distance: int) -> set[tuple[str, str]]:
    if not phashes or maximum_distance < 0:
        return set()
    bit_lengths = {_valid_phash(value)[1] for value in phashes.values() if _valid_phash(value)}
    if len(bit_lengths) != 1:
        return set()
    bit_count = bit_lengths.pop()
    segment_count = maximum_distance + 1
    if segment_count > bit_count:
        return set(combinations(sorted(phashes), 2))
    quotient, remainder = divmod(bit_count, segment_count)
    segment_ranges = []
    start = 0
    for index in range(segment_count):
        width = quotient + (1 if index < remainder else 0)
        segment_ranges.append((start, width))
        start += width

    buckets: dict[tuple[int, int], list[str]] = {}
    pairs: set[tuple[str, str]] = set()
    for record_path in sorted(phashes):
        parsed = _valid_phash(phashes[record_path])
        if parsed is None:
            continue
        value, _ = parsed
        for segment_index, (offset, width) in enumerate(segment_ranges):
            mask = (1 << width) - 1
            segment_value = (value >> offset) & mask
            bucket_key = (segment_index, segment_value)
            for other_path in buckets.get(bucket_key, []):
                pairs.add((other_path, record_path))
            buckets.setdefault(bucket_key, []).append(record_path)
    return pairs


def _connected_components(edges: list[dict]) -> list[set[str]]:
    adjacency: dict[str, set[str]] = {}
    for edge in edges:
        left, right = edge['left'], edge['right']
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)

    components = []
    remaining = set(adjacency)
    while remaining:
        start = min(remaining)
        stack = [start]
        component = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(sorted(adjacency.get(current, ()), reverse=True))
        remaining -= component
        if len(component) > 1:
            components.append(component)
    return sorted(components, key=lambda values: tuple(sorted(values)))


def _group_id(method: str, members: list[str]) -> str:
    material = method + '\0' + '\0'.join(sorted(members))
    return f'{method}-{hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]}'


def _member(record_path: str, record: dict) -> dict:
    return {
        'record_id': record.get('record_id', record_path),
        'filename': record.get('filename', Path(record_path).name),
        'relative_path': record_path,
    }


def _build_exact_groups(records: dict[str, dict]) -> list[dict]:
    by_hash: dict[str, list[str]] = {}
    for relative_path, record in records.items():
        sha256 = _valid_sha256(record.get('sha256'))
        if sha256:
            by_hash.setdefault(sha256, []).append(relative_path)

    groups = []
    for sha256, member_paths in sorted(by_hash.items()):
        if len(member_paths) < 2:
            continue
        member_paths.sort()
        groups.append({
            'group_id': _group_id('exact', member_paths),
            'duplicate_type': 'exact',
            'match_method': 'sha256',
            'duplicate_confidence': 1.0,
            'sha256': sha256,
            'members': [_member(path, records[path]) for path in member_paths],
            'matches': [
                {
                    'left': left,
                    'right': right,
                    'match_method': 'sha256',
                    'distance': 0,
                    'similarity': 1.0,
                }
                for left, right in combinations(member_paths, 2)
            ],
        })
    return groups


def _build_phash_edges(
    records: dict[str, dict],
    threshold: int,
    exact_pairs: set[tuple[str, str]],
    candidate_distance: int,
) -> list[dict]:
    phashes = {
        path: record['phash']
        for path, record in records.items()
        if _valid_phash(record.get('phash')) is not None
    }
    candidates = _candidate_phash_pairs(phashes, candidate_distance)
    edges = []
    for left, right in sorted(candidates):
        if (left, right) in exact_pairs:
            continue
        distance = _phash_distance(phashes[left], phashes[right])
        if distance is not None and distance <= threshold:
            edges.append({
                'left': left,
                'right': right,
                'match_method': 'phash',
                'distance': distance,
                'similarity': None,
            })
    return edges


def _build_visual_edges(
    records: dict[str, dict],
    candidate_pairs: set[tuple[str, str]],
    threshold: float,
    near_pairs: set[tuple[str, str]],
) -> list[dict]:
    embeddings: dict[str, np.ndarray] = {}
    pairs_by_dimension: dict[int, list[tuple[str, str]]] = {}
    left_vectors_by_dimension: dict[int, list[np.ndarray]] = {}
    right_vectors_by_dimension: dict[int, list[np.ndarray]] = {}
    for left, right in sorted(candidate_pairs):
        if (left, right) in near_pairs:
            continue
        if _valid_sha256(records[left].get('sha256')) and (
            _valid_sha256(records[left].get('sha256')) == _valid_sha256(records[right].get('sha256'))
        ):
            continue

        if left not in embeddings:
            embeddings[left] = _record_embedding(records[left])
        if right not in embeddings:
            embeddings[right] = _record_embedding(records[right])
        left_vector = embeddings[left]
        right_vector = embeddings[right]
        if left_vector is None or right_vector is None or left_vector.size != right_vector.size:
            continue
        dimension = left_vector.size
        pairs_by_dimension.setdefault(dimension, []).append((left, right))
        left_vectors_by_dimension.setdefault(dimension, []).append(left_vector)
        right_vectors_by_dimension.setdefault(dimension, []).append(right_vector)

    if not pairs_by_dimension:
        return []
    edges = []
    for dimension, pair_paths in pairs_by_dimension.items():
        left_matrix = np.stack(left_vectors_by_dimension[dimension])
        right_matrix = np.stack(right_vectors_by_dimension[dimension])
        similarities = np.einsum('ij,ij->i', left_matrix, right_matrix)
        for (left, right), similarity in zip(pair_paths, similarities, strict=True):
            score = float(np.clip(similarity, -1.0, 1.0))
            if score >= threshold:
                edges.append({
                    'left': left,
                    'right': right,
                    'match_method': 'openclip',
                    'distance': None,
                    'similarity': round(score, 6),
                })
    return edges


def _groups_from_edges(
    records: dict[str, dict],
    edges: list[dict],
    duplicate_type: str,
    confidence_for_edge,
) -> list[dict]:
    groups = []
    for member_paths_set in _connected_components(edges):
        member_paths = sorted(member_paths_set)
        group_edges = [
            edge for edge in edges
            if edge['left'] in member_paths_set and edge['right'] in member_paths_set
        ]
        confidences = [confidence_for_edge(edge) for edge in group_edges]
        groups.append({
            'group_id': _group_id(duplicate_type, member_paths),
            'duplicate_type': duplicate_type,
            'match_method': group_edges[0]['match_method'],
            'duplicate_confidence': round(float(min(confidences)), 6) if confidences else None,
            'members': [_member(path, records[path]) for path in member_paths],
            'matches': group_edges,
        })
    return groups


def _fingerprint(records: dict[str, dict], library_root: str, options: dict) -> str:
    rows = []
    for relative_path, record in sorted(records.items()):
        embedding_signature = None
        embedding_path = record.get('embedding_path')
        if record.get('embedding_status') == 'completed' and embedding_path:
            try:
                stat = Path(embedding_path).stat()
                embedding_signature = (str(Path(embedding_path).resolve()), stat.st_size, stat.st_mtime_ns)
            except OSError:
                embedding_signature = (str(embedding_path), None, None)
        rows.append((
            relative_path,
            record.get('file_status'),
            _record_image_path(library_root, record).is_file(),
            record.get('sha256'),
            record.get('phash'),
            embedding_signature,
        ))
    body = json.dumps({'version': DUPLICATE_VERSION, 'options': options, 'records': rows}, separators=(',', ':'))
    return hashlib.sha256(body.encode('utf-8')).hexdigest()


def _reset_record_duplicate_fields(record: dict) -> None:
    record['duplicate_group_id'] = None
    record['duplicate_type'] = 'none'
    record['duplicate_confidence'] = None
    record['duplicate_match_method'] = None
    record['duplicate_match_distance'] = None
    record['duplicate_updated_at'] = None
    record['duplicate_groups'] = []


def _apply_groups(records: dict[str, dict], groups: list[dict], updated_at: str) -> None:
    priority = {'exact': 0, 'near': 1, 'visual': 2}
    groups_by_member: dict[str, list[dict]] = {}
    for group in groups:
        for member in group['members']:
            groups_by_member.setdefault(member['relative_path'], []).append(group)

    for relative_path, record in records.items():
        _reset_record_duplicate_fields(record)
        member_groups = sorted(
            groups_by_member.get(relative_path, []),
            key=lambda group: (priority[group['duplicate_type']], group['group_id']),
        )
        record['duplicate_groups'] = [
            {
                'group_id': group['group_id'],
                'duplicate_type': group['duplicate_type'],
                'match_method': group['match_method'],
                'duplicate_confidence': group['duplicate_confidence'],
            }
            for group in member_groups
        ]
        if member_groups:
            group = member_groups[0]
            record['duplicate_group_id'] = group['group_id']
            record['duplicate_type'] = group['duplicate_type']
            record['duplicate_confidence'] = group['duplicate_confidence']
            record['duplicate_match_method'] = group['match_method']
            relevant_matches = [
                match for match in group['matches']
                if relative_path in (match['left'], match['right'])
            ]
            if relevant_matches:
                match = min(
                    relevant_matches,
                    key=lambda entry: (
                        entry['distance'] if entry['distance'] is not None else 0,
                        -(entry['similarity'] if entry['similarity'] is not None else 1.0),
                    ),
                )
                record['duplicate_match_distance'] = (
                    match['distance'] if match['distance'] is not None else match['similarity']
                )
            record['duplicate_updated_at'] = updated_at


def _build_response(library_id: str, library_root: str, summary: dict, processed: int, skipped: int) -> dict:
    groups = summary.get('groups', [])
    exact_groups = [group for group in groups if group['duplicate_type'] == 'exact']
    near_groups = [group for group in groups if group['duplicate_type'] == 'near']
    visual_groups = [group for group in groups if group['duplicate_type'] == 'visual']
    duplicates_found = sum(max(0, len(group['members']) - 1) for group in groups)
    return {
        'library_id': library_id,
        'library_root': library_root,
        'total_records': summary.get('total_records', 0),
        'exact_groups': len(exact_groups),
        'near_duplicate_groups': len(near_groups),
        'visual_matches': sum(len(group['matches']) for group in visual_groups),
        'visual_groups': len(visual_groups),
        'duplicates_found': duplicates_found,
        'processed': processed,
        'skipped': skipped,
        'duplicate_version': DUPLICATE_VERSION,
        'groups': groups,
    }


@serialize_index_update
def process_library_duplicates(
    folder_path: str | None = None,
    force: bool = False,
    include_visual: bool = True,
    phash_threshold: int = PHASH_HAMMING_THRESHOLD,
    openclip_threshold: float = OPENCLIP_DUPLICATE_THRESHOLD,
    visual_candidate_phash_distance: int = VISUAL_CANDIDATE_PHASH_DISTANCE,
) -> dict:
    if phash_threshold < 0 or phash_threshold > 256:
        raise ValueError('pHash threshold must be between 0 and 256.')
    if not -1 <= openclip_threshold <= 1:
        raise ValueError('OpenCLIP cosine threshold must be between -1 and 1.')
    if visual_candidate_phash_distance < phash_threshold or visual_candidate_phash_distance > 256:
        raise ValueError('Visual candidate pHash distance must be at least the pHash threshold and at most 256.')

    normalized_root = validate_folder_path(folder_path) if folder_path else None
    index = _load_index()
    library_items = []
    for library_id, library in index['libraries'].items():
        library_root = str(library.get('library_root') or '')
        if normalized_root and Path(library_root).resolve() != Path(normalized_root).resolve():
            continue
        if not isinstance(library.get('records'), dict):
            continue
        library_items.append((library_id, library_root, library))
    if normalized_root and not library_items:
        raise ValueError('Library has not been scanned yet.')

    options = {
        'include_visual': include_visual,
        'phash_threshold': phash_threshold,
        'openclip_threshold': openclip_threshold,
        'visual_candidate_phash_distance': visual_candidate_phash_distance,
    }
    responses = []
    for library_id, library_root, library in library_items:
        records = library['records']
        for relative_path, record in records.items():
            if not isinstance(record, dict) or record.get('file_status') == 'missing':
                continue
            image_path = _record_image_path(library_root, record)
            if not image_path.is_file():
                continue
            if _valid_sha256(record.get('sha256')) is None:
                record['sha256'] = _compute_sha256(image_path)
        fingerprint = _fingerprint(records, library_root, options)
        cached_summary = library.get('duplicate_detection')
        usable_paths = [
            relative_path for relative_path, record in records.items()
            if isinstance(record, dict)
            and record.get('file_status') != 'missing'
            and _record_image_path(library_root, record).is_file()
        ]
        if (
            not force
            and isinstance(cached_summary, dict)
            and cached_summary.get('version') == DUPLICATE_VERSION
            and cached_summary.get('fingerprint') == fingerprint
        ):
            responses.append(_build_response(
                library_id, library_root, cached_summary, processed=0, skipped=len(usable_paths)
            ))
            continue

        eligible_records = {
            relative_path: record
            for relative_path, record in records.items()
            if isinstance(record, dict)
            and record.get('file_status') != 'missing'
            and _record_image_path(library_root, record).is_file()
        }
        exact_groups = _build_exact_groups(eligible_records)
        exact_pairs = {
            tuple(sorted((match['left'], match['right'])))
            for group in exact_groups
            for match in group['matches']
        }
        phash_edges = _build_phash_edges(
            eligible_records,
            phash_threshold,
            exact_pairs,
            visual_candidate_phash_distance if include_visual else phash_threshold,
        )
        phash_candidate_pairs = {
            tuple(sorted((edge['left'], edge['right'])))
            for edge in _build_phash_edges(
                eligible_records,
                visual_candidate_phash_distance,
                exact_pairs,
                visual_candidate_phash_distance,
            )
        } if include_visual else set()
        near_pairs = {
            tuple(sorted((edge['left'], edge['right'])))
            for edge in phash_edges
        }
        near_groups = _groups_from_edges(
            eligible_records,
            phash_edges,
            'near',
            lambda edge: 1.0 - edge['distance'] / max(
                _valid_phash(eligible_records[edge['left']].get('phash'))[1], 1
            ),
        )

        visual_edges = _build_visual_edges(
            eligible_records,
            phash_candidate_pairs,
            openclip_threshold,
            near_pairs,
        ) if include_visual else []
        visual_groups = _groups_from_edges(
            eligible_records,
            visual_edges,
            'visual',
            lambda edge: edge['similarity'],
        )
        groups = sorted(
            exact_groups + near_groups + visual_groups,
            key=lambda group: (
                {'exact': 0, 'near': 1, 'visual': 2}[group['duplicate_type']],
                group['group_id'],
            ),
        )
        updated_at = datetime.now(timezone.utc).isoformat(timespec='seconds')
        _apply_groups(records, groups, updated_at)
        summary = {
            'version': DUPLICATE_VERSION,
            'fingerprint': fingerprint,
            'updated_at': updated_at,
            'options': options,
            'total_records': len(records),
            'groups': groups,
        }
        library['duplicate_detection'] = summary
        responses.append(_build_response(
            library_id, library_root, summary, processed=len(eligible_records), skipped=0
        ))

    _save_index(index)
    if normalized_root:
        return responses[0]
    combined = {
        'total_records': sum(result['total_records'] for result in responses),
        'exact_groups': sum(result['exact_groups'] for result in responses),
        'near_duplicate_groups': sum(result['near_duplicate_groups'] for result in responses),
        'visual_matches': sum(result['visual_matches'] for result in responses),
        'visual_groups': sum(result['visual_groups'] for result in responses),
        'duplicates_found': sum(result['duplicates_found'] for result in responses),
        'processed': sum(result['processed'] for result in responses),
        'skipped': sum(result['skipped'] for result in responses),
        'duplicate_version': DUPLICATE_VERSION,
        'libraries': responses,
    }
    return combined


def list_library_duplicates(folder_path: str | None = None) -> dict:
    normalized_root = validate_folder_path(folder_path) if folder_path else None
    index = _load_index()
    summaries = []
    for library_id, library in index['libraries'].items():
        library_root = str(library.get('library_root') or '')
        if normalized_root and Path(library_root).resolve() != Path(normalized_root).resolve():
            continue
        summary = library.get('duplicate_detection')
        if not isinstance(summary, dict):
            continue
        summaries.append(_build_response(
            library_id,
            library_root,
            summary,
            processed=0,
            skipped=summary.get('total_records', 0),
        ))
    if normalized_root:
        if not summaries:
            raise ValueError('Duplicate detection has not been run for this library.')
        return summaries[0]
    return {
        'total_records': sum(item['total_records'] for item in summaries),
        'exact_groups': sum(item['exact_groups'] for item in summaries),
        'near_duplicate_groups': sum(item['near_duplicate_groups'] for item in summaries),
        'visual_matches': sum(item['visual_matches'] for item in summaries),
        'visual_groups': sum(item['visual_groups'] for item in summaries),
        'duplicates_found': sum(item['duplicates_found'] for item in summaries),
        'libraries': summaries,
    }
