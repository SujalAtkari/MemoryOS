import io
import json
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from app.services import embedding_service, search_service


RECORD_COUNT = 5000
QUERY = 'government passport document'


def main() -> None:
    query_started = time.perf_counter()
    query_vector = embedding_service.encode_text_embeddings([QUERY])[0]
    query_embedding_seconds = time.perf_counter() - query_started

    generator = np.random.default_rng(20261001)
    vectors = generator.normal(size=(RECORD_COUNT, query_vector.size)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors[0] = query_vector

    similarity_started = time.perf_counter()
    similarities = vectors @ query_vector
    vector_similarity_seconds = time.perf_counter() - similarity_started

    with tempfile.TemporaryDirectory(prefix='memoryos-phase9-search-') as temporary:
        temporary_path = Path(temporary)
        library_path = temporary_path / 'synthetic-library'
        embedding_directory = temporary_path / 'embeddings'
        library_path.mkdir()
        embedding_directory.mkdir()

        image_buffer = io.BytesIO()
        Image.new('RGB', (1, 1), color=(80, 120, 160)).save(image_buffer, format='PNG')
        synthetic_image = image_buffer.getvalue()

        records = {}
        for index in range(RECORD_COUNT):
            relative_path = f'synthetic-{index:05d}.png'
            image_path = library_path / relative_path
            image_path.write_bytes(synthetic_image)
            embedding_path = embedding_directory / f'{index:05d}.npy'
            np.save(embedding_path, vectors[index], allow_pickle=False)
            records[relative_path] = {
                'record_id': relative_path,
                'relative_path': relative_path,
                'filename': relative_path,
                'extension': '.png',
                'file_status': 'available',
                'ai_processing_status': 'completed',
                'embedding_status': 'completed',
                'embedding_path': str(embedding_path),
                'ocr_text': 'synthetic government document record',
                'caption': 'synthetic document image record',
                'category': 'Notes & Documents',
                'width': 1,
                'height': 1,
            }

        index_path = temporary_path / 'library_index.json'
        index_path.write_text(json.dumps({
            'version': 1,
            'libraries': {
                'synthetic-library': {
                    'library_root': str(library_path),
                    'records': records,
                },
            },
        }), encoding='utf-8')

        original_encoder = search_service.encode_text_embeddings
        search_service.encode_text_embeddings = embedding_service.encode_text_embeddings
        original_token_score = search_service._token_score
        lexical_seconds = 0.0

        def measured_token_score(*args, **kwargs):
            nonlocal lexical_seconds
            started = time.perf_counter()
            result = original_token_score(*args, **kwargs)
            lexical_seconds += time.perf_counter() - started
            return result

        search_service.INDEX_PATH = index_path
        search_service._token_score = measured_token_score
        try:
            search_started = time.perf_counter()
            result = search_service.search_images(
                QUERY,
                folder_path=str(library_path),
                limit=20,
            )
            total_search_seconds = time.perf_counter() - search_started
            repeat_started = time.perf_counter()
            repeated_result = search_service.search_images(
                QUERY,
                folder_path=str(library_path),
                limit=20,
            )
            repeated_search_seconds = time.perf_counter() - repeat_started
        finally:
            search_service.encode_text_embeddings = original_encoder
            search_service._token_score = original_token_score

    if not result['results'] or result['results'][0]['record_id'] != 'synthetic-00000.png':
        raise RuntimeError('Synthetic benchmark failed its known nearest-result assertion.')
    if [item['record_id'] for item in result['results']] != [
        item['record_id'] for item in repeated_result['results']
    ]:
        raise RuntimeError('Repeated synthetic search returned a different ranking.')

    print(json.dumps({
        'dataset': 'synthetic metadata, placeholder image files, and random normalized embeddings',
        'record_count': RECORD_COUNT,
        'embedding_dimension': int(query_vector.size),
        'real_openclip_query_embedding_seconds': round(query_embedding_seconds, 6),
        'numpy_vector_similarity_seconds': round(vector_similarity_seconds, 6),
        'search_lexical_token_scoring_seconds': round(lexical_seconds, 6),
        'search_total_seconds_including_cached_query_embedding': round(total_search_seconds, 6),
        'repeated_search_total_seconds': round(repeated_search_seconds, 6),
        'results_returned': result['returned_results'],
        'top_record_id': result['results'][0]['record_id'],
        'top_semantic_score': result['results'][0]['semantic_score'],
        'top_hybrid_score': result['results'][0]['hybrid_score'],
        'known_top_result_assertion': 'passed',
        'synthetic_vectors_are_not_real_image_ai_outputs': True,
    }, indent=2))


if __name__ == '__main__':
    main()
