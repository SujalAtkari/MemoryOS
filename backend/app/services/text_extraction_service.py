import re

_TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9'/-]*")
_STOP_WORDS = frozenset({
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'in',
    'is', 'it', 'of', 'on', 'or', 'the', 'this', 'to', 'was', 'with',
    'date', 'number',
})


def extract_keywords_and_phrases(*values: object) -> tuple[list[str], list[str]]:
    tokens = [token for value in values for token in _TOKEN_PATTERN.findall(str(value or '').lower())]
    meaningful = [token for token in tokens if token not in _STOP_WORDS and len(token) > 1]
    keywords = sorted(set(meaningful))
    phrases = sorted({
        f'{meaningful[index]} {meaningful[index + 1]}'
        for index in range(len(meaningful) - 1)
        if meaningful[index] != meaningful[index + 1]
    })
    return keywords, phrases