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
from pipeline.predictor import predict, warmup
from utils import validate_response

HOST = '0.0.0.0'
PORT = int(os.environ.get('PORT', 9053))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app):
    # Load weights and run dummy inferences before the first real frame: there
    # is no timing allowance for a slow first request.
    warmup()
    yield


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
    return {
        'service': 'drone-flyby-usecase',
        'uptime': '{}'.format(datetime.timedelta(seconds=time.time() - start_time)),
    }


@app.get('/')
def index():
    return "Your endpoint is running!"


if __name__ == '__main__':
    uvicorn.run('api:app', host=HOST, port=PORT)
