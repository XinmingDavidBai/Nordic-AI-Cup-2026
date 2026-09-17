"""One request in, one response out. This is what api.py calls.

Per frame:

    1. decode the view
    2. ego-motion: how far the ground moved since our previous request
    3. detector on the view -> source-pixel detections
    4. tracker: move every known object, match, update, forget
    5. camera memory: age + move the coverage map, mark the current view seen
    6. answer: every live track, whole frame, frame-global boxes
    7. camera: best legal next view

Every stage is guarded. Whatever breaks, a valid response still goes out: an
exception loses the frame, an empty answer only loses recall.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from dtos import (
    DroneFlybyPredictionDto,
    DroneFlybyPredictRequestDto,
    DroneFlybyPredictResponseDto,
)
from pipeline import config, recorder
from pipeline.camera_policy import CameraPolicy
from pipeline.detector import Detector, build_detector
from pipeline.egomotion import EgoMotionEstimator
from pipeline.tracker import Tracker
from utils import clip_bbox_to_frame, decode_view, source_bbox_to_global

logger = logging.getLogger(__name__)


@dataclass
class SequenceState:
    sequence_id: str
    ego: EgoMotionEstimator = field(default_factory=EgoMotionEstimator)
    tracker: Tracker = field(default_factory=Tracker)
    policy: CameraPolicy = field(default_factory=CameraPolicy)
    last_frame_index: Optional[int] = None


_lock = threading.Lock()
_detector: Optional[Detector] = None
_states: Dict[str, SequenceState] = {}
# Last per-frame debug info, read by debug_replay.py.
last_debug: dict = {}


def get_detector() -> Detector:
    global _detector
    if _detector is None:
        _detector = build_detector()
    return _detector


def warmup() -> None:
    """Load the model and run a couple of dummy inferences before traffic arrives."""
    started = time.perf_counter()
    get_detector().warmup()
    logger.info('Warmup done in %.0f ms (detector: %s)', (time.perf_counter() - started) * 1e3, get_detector().name)


def reset() -> None:
    """Forget all sequence state (used between local runs)."""
    with _lock:
        _states.clear()


def _state_for(request: DroneFlybyPredictRequestDto) -> SequenceState:
    state = _states.get(request.sequence_id)
    restarted = (
        state is not None
        and state.last_frame_index is not None
        and request.frame_index <= state.last_frame_index
    )
    if state is None or restarted:
        if restarted:
            logger.info('Sequence %s restarted, resetting state', request.sequence_id)
        # Only one attempt runs at a time; drop older sequences to bound memory.
        _states.clear()
        state = SequenceState(request.sequence_id)
        _states[request.sequence_id] = state
    return state


def predict(request: DroneFlybyPredictRequestDto) -> DroneFlybyPredictResponseDto:
    with _lock:
        return _predict_locked(request)


def _predict_locked(request: DroneFlybyPredictRequestDto) -> DroneFlybyPredictResponseDto:
    timings = {}
    t0 = time.perf_counter()

    if request.camera_command_feedback is not None:
        feedback = request.camera_command_feedback
        logger.warning('Camera command from frame %s was ignored: %s', feedback.frame, feedback.reason)

    annotations: List[DroneFlybyPredictionDto] = []
    requested_view = None
    debug = {'frame': request.frame, 'frame_index': request.frame_index}

    try:
        state = _state_for(request)
        view = request.view
        region = view.source_region_xyxy
        level = view.resolution_level
        gap = 1 if state.last_frame_index is None else max(1, request.frame_index - state.last_frame_index)
        state.last_frame_index = request.frame_index

        image = decode_view(view)
        timings['decode'] = time.perf_counter()

        try:
            state.ego.update(image, region, request.frame_index)
        except Exception:
            logger.exception('Ego-motion failed on frame %s (using the previous flow model)', request.frame)
        displacement = state.ego.displacement_at_center(gap)
        timings['ego'] = time.perf_counter()

        try:
            detections = get_detector().detect(image, request)
        except Exception:
            logger.exception('Detector failed on frame %s', request.frame)
            detections = []
        timings['detect'] = time.perf_counter()

        try:
            state.tracker.step(request.frame_index, state.ego, detections, region, level, gap)
            # Matched tracks are flow measurements too; they matter most when
            # consecutive views did not overlap enough to register.
            state.ego.add_samples(state.tracker.flow_samples, request.frame_index)
        except Exception:
            logger.exception('Tracker failed on frame %s', request.frame)
        timings['track'] = time.perf_counter()

        try:
            annotations = _to_annotations(state.tracker.outputs(request.frame_index), request)
        except Exception:
            logger.exception('Building annotations failed on frame %s', request.frame)
            annotations = []

        try:
            state.policy.advance(displacement, gap)
            state.policy.observe(region, level)
            requested_view = state.policy.choose(
                request, state.tracker, state.ego.velocity, request.frame_index
            )
        except Exception:
            logger.exception('Camera policy failed on frame %s', request.frame)
            requested_view = None
        timings['policy'] = time.perf_counter()

        debug.update(
            displacement=tuple(round(d, 1) for d in displacement),
            ego_ok=state.ego.last_measurement_ok,
            ego_response=round(state.ego.last_response, 3),
            ego_tiles=state.ego.last_tile_count,
            velocity=tuple(np.round(state.ego.velocity, 1)),
            flow_theta=np.round(state.ego.theta, 2).tolist(),
            detections=[(d.object_id, round(d.confidence, 3), tuple(round(c, 1) for c in d.box)) for d in detections],
            tracks=len(state.tracker.tracks),
            policy=dict(state.policy.last_choice_debug),
        )
    except Exception:
        logger.exception('Pipeline failed on frame %s; answering empty', request.frame)

    response = DroneFlybyPredictResponseDto(
        request_id=request.request_id,
        frame=request.frame,
        annotations=annotations,
        requested_view=requested_view,
    )

    end = time.perf_counter()
    stage_ms, previous = {}, t0
    for name in ('decode', 'ego', 'detect', 'track', 'policy'):
        if name in timings:
            stage_ms[name] = round((timings[name] - previous) * 1e3, 1)
            previous = timings[name]
    stage_ms['total'] = round((end - t0) * 1e3, 1)
    debug['timings_ms'] = stage_ms
    last_debug.clear()
    last_debug.update(debug)
    if config.LOG_TIMINGS:
        logger.info(
            'frame %s idx %s L%s: %d dets, %d tracks, %d answers, next=%s, ms=%s',
            request.frame, request.frame_index, request.view.resolution_level,
            len(debug.get('detections', [])), debug.get('tracks', 0), len(annotations),
            None if requested_view is None else (requested_view.resolution_level, requested_view.center_x, requested_view.center_y),
            stage_ms,
        )

    recorder.record(request, response)
    return response


def _to_annotations(answers, request) -> List[DroneFlybyPredictionDto]:
    out = []
    for object_id, confidence, box in answers:
        bbox = clip_bbox_to_frame(source_bbox_to_global(box, request.original_width, request.original_height))
        if bbox is None:
            continue
        out.append(
            DroneFlybyPredictionDto(
                object_id=object_id,
                bbox=[round(float(c), 6) for c in bbox],
                confidence=round(float(min(max(confidence, 0.0), 1.0)), 5),
            )
        )
    return out
