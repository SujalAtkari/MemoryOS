import json
import os
import tempfile
import threading
import time
from functools import wraps
from pathlib import Path
from typing import Callable, ParamSpec, TypeVar

P = ParamSpec('P')
R = TypeVar('R')

_index_update_lock = threading.RLock()


def serialize_index_update(function: Callable[P, R]) -> Callable[P, R]:
    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        with _index_update_lock:
            return function(*args, **kwargs)

    return wrapped


def save_json_atomically(index_path: Path, payload: dict) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f'.{index_path.name}.',
        suffix='.tmp',
        dir=index_path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())

        for attempt in range(6):
            try:
                os.replace(temporary_path, index_path)
                return
            except PermissionError as exc:
                winerror = getattr(exc, 'winerror', None)
                if attempt == 5 or (winerror not in (None, 5, 32) and exc.errno not in (13,)):
                    raise
                time.sleep(0.025 * (attempt + 1))
    except (OSError, TypeError, ValueError):
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
