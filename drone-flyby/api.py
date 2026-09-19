"""The endpoint the evaluation service calls.

You should not need to change much in here: the model, tracking and camera
logic live in ``pipeline/`` (see PIPELINE.md). Leave the transport alone.

The URL you submit is used exactly as you give it, path included, so if you
keep the ``/predict`` route below then submit ``http://<your-host>:9053/predict``
rather than just the host.
"""

import datetime
import logging
import os
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from dtos import DroneFlybyPredictRequestDto, DroneFlybyPredictResponseDto
# The full pipeline (detector + tracker + camera policy). The organisers'
# baseline is still in example.py; swap this import back to compare.
from pipeline import config
from pipeline.predictor import detector_info, predict, sequence_info, warmup
from utils import validate_response

HOST = '0.0.0.0'
PORT = int(os.environ.get('PORT', 9053))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# The server records every request unless RECORD_DIR is set (even to empty, which
# turns it off): a validation run you did not record cannot be debugged afterwards.
if 'RECORD_DIR' not in os.environ:
    config.RECORD_DIR = str(config.ROOT / 'recordings')


@asynccontextmanager
async def lifespan(_app):
    # Load weights and run dummy inferences before the first real frame: there
    # is no timing allowance for a slow first request.
    warmup()
    check_detector()
    logger.info('Process %s serving on port %s; recording to %s', os.getpid(), PORT,
                config.RECORD_DIR or '(off)')
    yield


def check_detector():
    """Refuse to serve without a working YOLO detector.

    Anything else (gt, none, edges, or auto falling back to none because the
    weights are missing) answers every frame with a valid, empty response, so
    the evaluator sees a healthy server scoring zero.
    """
    info = detector_info()
    problem = None
    if info['backend'] != 'yolo':
        problem = (
            f"detector resolved to {info['backend']!r}, not 'yolo' "
            f'(DETECTOR_BACKEND={config.DETECTOR_BACKEND!r}, DETECTOR_WEIGHTS={config.DETECTOR_WEIGHTS})'
        )
    elif info['warmup_error']:
        device = info['device_info']
        problem = (f"YOLO warmup inference failed on device {device['device']!r} "
                   f"({device['how']}; DETECTOR_DEVICE={config.DETECTOR_DEVICE!r}): {info['warmup_error']}")
    if problem is None:
        device = info['device_info']
        if config.DETECTOR_REQUIRE_GPU and device['kind'] == 'cpu':
            raise RuntimeError(
                f"Refusing to start: DETECTOR_REQUIRE_GPU is set but the detector is on the CPU "
                f"({device['how']}: {device['reason']})."
            )
        where = device['kind'].upper() + (f" {device['gpu_name']}" if device['gpu_name'] else '')
        logger.info(
            'Detector OK: yolo, weights %s (sha256 %s), on %s (%s: %s)',
            info['weights_path'], info['weights_sha256'], where, device['how'], device['reason'],
        )
        return
    if config.ALLOW_NON_YOLO_DETECTOR:
        logger.warning('!!! %s. Starting anyway because ALLOW_NON_YOLO_DETECTOR is set: '
                       'NEVER validate or submit against this server.', problem)
        return
    raise RuntimeError(
        f'Refusing to start: {problem}. A server like this answers every real frame '
        'with nothing. Fix DETECTOR_BACKEND/DETECTOR_WEIGHTS/DETECTOR_DEVICE, or set '
        'ALLOW_NON_YOLO_DETECTOR=1 for local debugging only.'
    )


app = FastAPI(lifespan=lifespan)
start_time = time.time()


@app.post('/predict', response_model=DroneFlybyPredictResponseDto)
def predict_endpoint(request: DroneFlybyPredictRequestDto):
    """Answer one frame."""
    response = predict(request)

    # Fail here, loudly, rather than having the evaluator silently discard the
    # frame. Every rule this checks is a rule the evaluator also enforces.
    validate_response(response)

    logger.info(
        'frame %s (index %s) L%s at (%s, %s): returned %s detections',
        request.frame,
        request.frame_index,
        request.view.resolution_level,
        request.view.center_x,
        request.view.center_y,
        len(response.annotations),
    )
    return response


@app.get('/api')
def hello():
    """Check this before spending a validation attempt: detector.backend must be
    'yolo' and detector.weights_sha256 must match the checkpoint you meant to deploy.
    Several calls must all show the same process.pid: different ones mean several
    processes answer, and each would track the run separately."""
    return {
        'service': 'drone-flyby-usecase',
        'uptime': '{}'.format(datetime.timedelta(seconds=time.time() - start_time)),
        'process': {
            'pid': os.getpid(),
            'started_at': datetime.datetime.fromtimestamp(start_time, datetime.timezone.utc).isoformat(),
            'recording_to': config.RECORD_DIR or None,
        },
        'detector': detector_info(),
        'sequences': sequence_info(),
    }


@app.get('/')
def index():
    return "Your endpoint is running!"


if __name__ == '__main__':
    uvicorn.run('api:app', host=HOST, port=PORT)
