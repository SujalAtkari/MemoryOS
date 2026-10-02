import json
from io import BytesIO
from urllib.error import URLError

from app.services import local_llm_service


def ollama_response(ranking):
    return BytesIO(json.dumps({
        'message': {'content': json.dumps({'ranking': ranking})},
    }).encode('utf-8'))


def accept_request(request, timeout):
    assert request.full_url.endswith('/api/chat')
    assert timeout == local_llm_service.RERANK_TIMEOUT_SECONDS
    return ollama_response(['1', 'unknown', '1', '0'])


def reject_request(request, timeout):
    raise URLError(f'{request.full_url} did not respond within {timeout}s')


def test_reranker_accepts_only_supplied_candidate_ids(monkeypatch):
    monkeypatch.setattr(local_llm_service, 'OLLAMA_MODEL', 'test-model')
    monkeypatch.setattr(local_llm_service, 'urlopen', accept_request)

    result = local_llm_service.rerank_candidates(
        'find the puppy',
        [{'record': {'filename': 'cat.jpg'}}, {'record': {'filename': 'puppy.jpg'}}],
    )

    assert result == {
        'status': 'used',
        'model': 'test-model',
        'order': ['1', '0'],
    }


def test_reranker_reports_unavailable_local_ollama(monkeypatch):
    monkeypatch.setattr(
        local_llm_service,
        'urlopen',
        reject_request,
    )

    result = local_llm_service.rerank_candidates(
        'find the puppy',
        [{'record': {'filename': 'puppy.jpg'}}],
    )

    assert result['status'] == 'unavailable'
    assert 'Ollama is unavailable' in result['detail']


def test_reranker_skips_empty_candidate_sets():
    assert local_llm_service.rerank_candidates('any query', []) == {
        'status': 'skipped',
        'model': local_llm_service.OLLAMA_MODEL,
        'order': [],
    }
