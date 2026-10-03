import json
import os
from collections import Counter
from pathlib import Path

from app.services.library_service import INDEX_PATH, validate_folder_path

CATEGORIES = (
    'Government & Identity',
    'Education',
    'Medical & Health',
    'Finance',
    'Bills & Receipts',
    'Work & Professional',
    'Travel',
    'Events & Celebrations',
    'People & Family',
    'Nature & Places',
    'Animals & Pets',
    'Food & Drinks',
    'Screenshots',
    'Notes & Documents',
    'Others',
)
AI_STATUSES = ('completed', 'pending', 'processing', 'failed')
CLASSIFICATION_STATUSES = ('classified', 'needs_review', 'processing', 'pending', 'failed')
FILE_STATUSES = ('available', 'missing', 'modified', 'unverified')
DUPLICATE_TYPES = ('exact', 'near', 'visual')


def _empty_stats() -> dict:
    return {
        'total_images': 0,
        'processed_images': 0,
        'invalid_records': 0,
        'processing': {status: 0 for status in AI_STATUSES} | {'other': 0},
        'classification': {status: 0 for status in CLASSIFICATION_STATUSES} | {'other': 0},
        'file_status': {status: 0 for status in FILE_STATUSES},
        'categories': {
            'total_classified': 0,
            'categorized_images': 0,
            'classified_without_category': 0,
            'distribution': [
                {'category': category, 'count': 0, 'percentage': 0.0}
                for category in CATEGORIES
            ],
        },
        'importance': {status: 0 for status in ('critical', 'important', 'normal', 'low')},
        'protected_images': 0,
        'lifecycle': {status: 0 for status in ('keep', 'review', 'archive', 'delete')},
        'duplicates': {
            'groups': 0,
            'exact_groups': 0,
            'near_duplicate_groups': 0,
            'visual_duplicate_groups': 0,
            'records_in_groups': 0,
        },
        'library_count': 0,
        'invalid_libraries': 0,
        'libraries': [],
        'recent_images': [],
    }


def _read_index() -> dict:
    if not INDEX_PATH.exists():
        return {'libraries': {}}
    try:
        with INDEX_PATH.open('r', encoding='utf-8') as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError('The local library index could not be read or is malformed.') from exc
    if not isinstance(payload, dict) or not isinstance(payload.get('libraries'), dict):
        raise RuntimeError('The local library index has an invalid structure.')
    return payload


def _safe_library_name(library_root: object, library_id: str) -> str:
    if isinstance(library_root, str) and library_root.strip():
        name = Path(library_root).name
        if name:
            return name
    return library_id


def _duplicate_groups(library: dict) -> dict[str, dict]:
    groups_by_id: dict[str, dict] = {}
    summary = library.get('duplicate_detection')
    source_groups = summary.get('groups') if isinstance(summary, dict) else None
    if isinstance(source_groups, list):
        for group in source_groups:
            if not isinstance(group, dict):
                continue
            group_id = group.get('group_id')
            duplicate_type = group.get('duplicate_type')
            members = group.get('members')
            if not isinstance(group_id, str) or duplicate_type not in DUPLICATE_TYPES:
                continue
            member_paths = {
                member.get('relative_path')
                for member in members if isinstance(member, dict)
                if isinstance(member.get('relative_path'), str)
            } if isinstance(members, list) else set()
            if len(member_paths) < 2:
                continue
            groups_by_id.setdefault(group_id, {
                'duplicate_type': duplicate_type,
                'members': member_paths,
            })
        return groups_by_id

    records = library.get('records')
    if not isinstance(records, dict):
        return groups_by_id
    for relative_path, record in records.items():
        if not isinstance(relative_path, str) or not isinstance(record, dict):
            continue
        memberships = record.get('duplicate_groups')
        if not isinstance(memberships, list):
            continue
        for membership in memberships:
            if not isinstance(membership, dict):
                continue
            group_id = membership.get('group_id')
            duplicate_type = membership.get('duplicate_type')
            if not isinstance(group_id, str) or duplicate_type not in DUPLICATE_TYPES:
                continue
            group = groups_by_id.setdefault(group_id, {
                'duplicate_type': duplicate_type,
                'members': set(),
            })
            group['members'].add(relative_path)
    return {
        group_id: group
        for group_id, group in groups_by_id.items()
        if len(group['members']) >= 2
    }


def _summarize_library(library_id: str, library: dict) -> dict:
    stats = _empty_stats()
    records = library.get('records')
    if not isinstance(records, dict):
        stats['invalid_records'] = 1
        records = {}

    category_counts = Counter()
    unique_members = set()
    for relative_path, record in records.items():
        if not isinstance(record, dict):
            stats['invalid_records'] += 1
            continue
        stats['total_images'] += 1

        ai_status = record.get('ai_processing_status')
        if ai_status not in AI_STATUSES:
            ai_status = 'pending' if ai_status is None else 'other'
        stats['processing'][ai_status] += 1

        classification_status = record.get('classification_status')
        if classification_status not in CLASSIFICATION_STATUSES:
            classification_status = 'pending' if classification_status is None else 'other'
        stats['classification'][classification_status] += 1

        file_status = record.get('file_status')
        if file_status not in ('available', 'missing', 'modified'):
            file_status = 'unverified'
        stats['file_status'][file_status] += 1

        importance_status = record.get('importance_status')
        if record.get('is_important') is True:
            stats['importance']['important'] += 1
        else:
            stats['importance'][importance_status if importance_status in stats['importance'] else 'normal'] += 1
        if record.get('protection_status') == 'protected':
            stats['protected_images'] += 1
        lifecycle_status = record.get('lifecycle_status')
        stats['lifecycle'][lifecycle_status if lifecycle_status in stats['lifecycle'] else 'keep'] += 1

        category = record.get('category')
        if classification_status == 'classified':
            stats['categories']['total_classified'] += 1
            if isinstance(category, str) and category in CATEGORIES:
                category_counts[category] += 1
            else:
                stats['categories']['classified_without_category'] += 1

        if isinstance(relative_path, str):
            identity = (library_id, relative_path)
        else:
            identity = (library_id, str(record.get('record_id') or stats['total_images']))
        unique_members.add(identity)

    distribution_denominator = sum(category_counts.values())
    stats['categories']['categorized_images'] = distribution_denominator
    stats['categories']['distribution'] = [
        {
            'category': category,
            'count': category_counts[category],
            'percentage': round(category_counts[category] * 100 / distribution_denominator, 2)
            if distribution_denominator else 0.0,
        }
        for category in CATEGORIES
    ]
    stats['processed_images'] = stats['processing']['completed']

    groups = _duplicate_groups(library)
    stats['duplicates']['groups'] = len(groups)
    stats['duplicates']['exact_groups'] = sum(
        group['duplicate_type'] == 'exact' for group in groups.values()
    )
    stats['duplicates']['near_duplicate_groups'] = sum(
        group['duplicate_type'] == 'near' for group in groups.values()
    )
    stats['duplicates']['visual_duplicate_groups'] = sum(
        group['duplicate_type'] == 'visual' for group in groups.values()
    )
    stats['duplicates']['records_in_groups'] = len({
        (library_id, member)
        for group in groups.values()
        for member in group['members']
    })

    return {
        'library_id': library_id,
        'display_name': _safe_library_name(library.get('library_root'), library_id),
        **stats,
    }


def get_dashboard_stats(folder_path: str | None = None) -> dict:
    normalized_root = validate_folder_path(folder_path) if folder_path else None
    index = _read_index()
    results = []
    invalid_libraries = 0
    for library_id, library in index['libraries'].items():
        if not isinstance(library, dict):
            invalid_libraries += 1
            continue
        library_root = library.get('library_root')
        if normalized_root:
            if not isinstance(library_root, str):
                continue
            try:
                matches = os.path.normcase(str(Path(library_root).resolve())) == os.path.normcase(normalized_root)
            except OSError:
                matches = False
            if not matches:
                continue
        results.append(_summarize_library(str(library_id), library))

    combined = _empty_stats()
    combined['invalid_libraries'] = invalid_libraries
    combined['library_count'] = len(results)
    category_totals = Counter()
    duplicate_member_ids = set()
    for library in results:
        for key in ('total_images', 'processed_images', 'invalid_records'):
            combined[key] += library[key]
        for key, count in library['processing'].items():
            combined['processing'][key] += count
        for key, count in library['classification'].items():
            combined['classification'][key] += count
        for key, count in library['file_status'].items():
            combined['file_status'][key] += count
        for key, count in library['importance'].items():
            combined['importance'][key] += count
        combined['protected_images'] += library['protected_images']
        for key, count in library['lifecycle'].items():
            combined['lifecycle'][key] += count
        combined['categories']['total_classified'] += library['categories']['total_classified']
        combined['categories']['classified_without_category'] += library['categories']['classified_without_category']
        for item in library['categories']['distribution']:
            category_totals[item['category']] += item['count']
        for key, count in library['duplicates'].items():
            if key != 'records_in_groups':
                combined['duplicates'][key] += count
        group_index = _duplicate_groups(
            index['libraries'][library['library_id']]
        )
        duplicate_member_ids.update(
            (library['library_id'], member)
            for group in group_index.values()
            for member in group['members']
        )

    categorized_images = sum(category_totals.values())
    combined['categories']['categorized_images'] = categorized_images
    combined['categories']['distribution'] = [
        {
            'category': category,
            'count': category_totals[category],
            'percentage': round(category_totals[category] * 100 / categorized_images, 2)
            if categorized_images else 0.0,
        }
        for category in CATEGORIES
    ]
    combined['duplicates']['records_in_groups'] = len(duplicate_member_ids)
    combined['libraries'] = [
        {
            'library_id': library['library_id'],
            'display_name': library['display_name'],
            'total_images': library['total_images'],
            'processed_images': library['processed_images'],
            'missing_files': library['file_status']['missing'],
        }
        for library in results
    ]
    recent_images = []
    for library_id, library in index['libraries'].items():
        if not isinstance(library, dict):
            continue
        library_root = library.get('library_root')
        if normalized_root:
            if not isinstance(library_root, str):
                continue
            try:
                matches = os.path.normcase(str(Path(library_root).resolve())) == os.path.normcase(normalized_root)
            except OSError:
                matches = False
            if not matches:
                continue
        records = library.get('records', {})
        if not isinstance(records, dict):
            continue
        for relative_path, record in records.items():
            if not isinstance(record, dict):
                continue
            recent_images.append({
                'library_id': str(library_id),
                'filename': record.get('filename') or Path(str(relative_path)).name,
                'relative_path': str(relative_path),
                'category': record.get('category') or 'Uncategorized',
                'modified_time': record.get('modified_time'),
                'ai_processing_status': record.get('ai_processing_status') or 'pending',
                'classification_status': record.get('classification_status') or 'pending',
                'file_status': record.get('file_status') or 'unverified',
            })
    combined['recent_images'] = sorted(
        recent_images,
        key=lambda item: item.get('modified_time') or '',
        reverse=True,
    )[:12]
    return combined
