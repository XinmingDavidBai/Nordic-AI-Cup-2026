"""Save what the evaluator sends, off the request thread.

With RECORD_DIR set, each request becomes:

    <RECORD_DIR>/<sequence_id>/<n>_frame_<frame>_idx_<frame_index>.png    the view
    <RECORD_DIR>/<sequence_id>/<n>_frame_<frame>_idx_<frame_index>.json   request (minus image) + response

``<n>`` is the arrival order in this process, so a repeated or out-of-order
request gets its own file instead of overwriting the first one, and the JSON
carries the arrival time and process id (several ids in one sequence means
several server processes answered it).

The competition rules allow recording the validation sequence, which makes this
the way to collect extra (unlabelled) training images with real crops.
"""

import base64
import itertools
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pipeline import config

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='recorder')
_arrivals = itertools.count(1)


def _safe(name: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in '-_.' else '_' for ch in name)[:80]


def record(request, response) -> None:
    if not config.RECORD_DIR:
        return
    try:
        request_json = request.model_dump()
        response_json = response.model_dump() if response is not None else None
    except Exception:
        logger.exception('Could not serialise request for recording')
        return
    meta = {'arrival': next(_arrivals), 'received_at': time.time(), 'pid': os.getpid()}
    _executor.submit(_write, request_json, response_json, meta)


def _write(request_json, response_json, meta) -> None:
    try:
        directory = Path(config.RECORD_DIR) / _safe(request_json['sequence_id'])
        directory.mkdir(parents=True, exist_ok=True)
        stem = f"{meta['arrival']:05d}_frame_{request_json['frame']:06d}_idx_{request_json['frame_index']:06d}"
        image_b64 = request_json['view'].pop('image')
        (directory / f'{stem}.png').write_bytes(base64.b64decode(image_b64))
        with open(directory / f'{stem}.json', 'w', encoding='utf-8') as handle:
            json.dump({'meta': meta, 'request': request_json, 'response': response_json}, handle)
    except Exception:
        logger.exception('Recording failed')
