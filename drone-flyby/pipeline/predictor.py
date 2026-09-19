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
from collections import OrderedDict
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
    frames: int = 0
    # Consecutive frames on which the detector raised / ran but found nothing.
    failure_streak: int = 0
    empty_streak: int = 0
    # Recent answers by request_id, so a retried request gets the same answer
    # instead of advancing (or wiping) the state a second time.
    responses: 'OrderedDict[str, DroneFlybyPredictResponseDto]' = field(default_factory=OrderedDict)

    def remember(self, request_id: str, response: DroneFlybyPredictResponseDto) -> None:
        self.responses[request_id] = response
        while len(self.responses) > _REMEMBERED_RESPONSES:
            self.responses.popitem(last=False)


_REMEMBERED_RESPONSES = 16

_lock = threading.Lock()
_detector: Optional[Detector] = None
# Most recently used last. A stray request for another sequence must not wipe
# the run in progress, so a few are kept.
_states: 'OrderedDict[str, SequenceState]' = OrderedDict()
# Totals since the server started, reported by /api.
_detector_stats = {'frames': 0, 'failed_frames': 0, 'empty_frames': 0, 'last_error': None}
_sequence_stats = {'new_sequences': 0, 'restarts': 0, 'repeated_requests': 0, 'stale_requests': 0, 'evicted': 0}
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


def detector_info() -> dict:
    """What detector is loaded plus how it has done so far (served on /api)."""
    with _lock:
        return {**get_detector().describe(), 'stats': dict(_detector_stats)}


def _alert_due(streak: int, threshold: int) -> bool:
    if threshold <= 0 or streak < threshold:
        return False
    repeat = max(1, config.DETECTOR_ALERT_REPEAT_FRAMES)
    return (streak - threshold) % repeat == 0


def _track_detector_health(state: SequenceState, request, detections, error: Optional[str]) -> None:
    _detector_stats['frames'] += 1
    if error is not None:
        _detector_stats['failed_frames'] += 1
        _detector_stats['last_error'] = error
        state.failure_streak += 1
        state.empty_streak = 0
        if _alert_due(state.failure_streak, config.DETECTOR_FAILURE_ALERT_FRAMES):
            logger.error(
                '!!! DETECTOR FAILED ON EVERY ONE OF THE LAST %d FRAMES (sequence %s, now frame %s, '
                'backend %s). Every answer is going out EMPTY, so this run scores ~0. Last error: %s',
                state.failure_streak, request.sequence_id, request.frame, get_detector().name, error,
            )
        return
    state.failure_streak = 0
    if detections:
        state.empty_streak = 0
        return
    _detector_stats['empty_frames'] += 1
    state.empty_streak += 1
    # The null detector finding nothing is expected; it only runs by explicit override.
    if get_detector().name != 'none' and _alert_due(state.empty_streak, config.DETECTOR_EMPTY_ALERT_FRAMES):
        logger.error(
            '!!! DETECTOR FOUND NOTHING ON EVERY ONE OF THE LAST %d FRAMES (sequence %s, now frame %s, '
            'backend %s). It is running without errors but detecting nothing: check the weights '
            '(GET /api) and DETECTOR_MIN_CONF.',
            state.empty_streak, request.sequence_id, request.frame, get_detector().name,
        )


def sequence_info() -> dict:
    """Which runs this process holds state for, and how requests arrived (served on /api)."""
    with _lock:
        return {
            'kept': [
                {'sequence_id': s.sequence_id, 'frames': s.frames, 'last_frame_index': s.last_frame_index}
                for s in _states.values()
            ],
            'stats': dict(_sequence_stats),
        }


def reset() -> None:
    """Forget all sequence state (used between local runs)."""
    with _lock:
        _states.clear()


def _route(request: DroneFlybyPredictRequestDto):
    """Find this request's run state and decide how to answer it.

    Returns (state, action): 'step' advances the pipeline, 'repeat' re-sends the
    answer already given to this request_id, 'stale' answers an older frame from
    the current tracker without advancing anything. State is only ever replaced
    when the same sequence visibly starts over at frame_index 0.
    """
    sequence_id = request.sequence_id
    state = _states.get(sequence_id)
    if state is None:
        _sequence_stats['new_sequences'] += 1
        if _states:
            logger.warning(
                'New sequence %s while holding state for %s; keeping those too (up to %d).',
                sequence_id, [s.sequence_id for s in _states.values()], config.SEQUENCE_STATES_KEPT,
            )
        state = _states[sequence_id] = SequenceState(sequence_id)
        while len(_states) > max(1, config.SEQUENCE_STATES_KEPT):
            evicted = _states.popitem(last=False)[1]
            _sequence_stats['evicted'] += 1
            logger.warning('Dropped state for least recently used sequence %s (%d frames).',
                           evicted.sequence_id, evicted.frames)
        return state, 'step'
    _states.move_to_end(sequence_id)

    if request.request_id in state.responses:
        _sequence_stats['repeated_requests'] += 1
        logger.warning(
            '!!! REPEATED REQUEST %s (sequence %s, frame_index %s): re-sending the earlier answer, '
            'state untouched. A client/proxy retry, or several requests racing.',
            request.request_id, sequence_id, request.frame_index,
        )
        return state, 'repeat'

    last = state.last_frame_index
    if last is None or request.frame_index > last:
        return state, 'step'
    if request.frame_index == 0 and last > 0:
        # A local evaluator rerun (always sequence 'local'), or a real restart.
        _sequence_stats['restarts'] += 1
        logger.warning('!!! Sequence %s started over at frame_index 0 (was at %s): new state.', sequence_id, last)
        state = _states[sequence_id] = SequenceState(sequence_id)
        return state, 'step'
    _sequence_stats['stale_requests'] += 1
    logger.warning(
        '!!! OUT-OF-ORDER REQUEST for sequence %s: frame_index %s after %s (request %s). '
        'Answering from the current tracker without advancing it.',
        sequence_id, request.frame_index, last, request.request_id,
    )
    return state, 'stale'


def predict(request: DroneFlybyPredictRequestDto) -> DroneFlybyPredictResponseDto:
    with _lock:
        try:
            state, action = _route(request)
        except Exception:
            logger.exception('Routing failed for request %s; treating it as a new sequence', request.request_id)
            state, action = SequenceState(request.sequence_id), 'step'
        if action == 'repeat':
            cached = state.responses[request.request_id]
            response = cached.model_copy()
        elif action == 'stale':
            response = _stale_response(state, request)
        else:
            response = _predict_locked(request, state)
        state.remember(request.request_id, response)
    recorder.record(request, response)
    return response


def _stale_response(state: SequenceState, request: DroneFlybyPredictRequestDto) -> DroneFlybyPredictResponseDto:
    """Best answer for an older frame: what the tracker currently believes, camera left alone."""
    annotations: List[DroneFlybyPredictionDto] = []
    try:
        annotations = _to_annotations(state.tracker.outputs(state.last_frame_index), request)
    except Exception:
        logger.exception('Building annotations for stale request %s failed', request.request_id)
    return DroneFlybyPredictResponseDto(
        request_id=request.request_id, frame=request.frame, annotations=annotations, requested_view=None,
    )


def _predict_locked(request: DroneFlybyPredictRequestDto, state: SequenceState) -> DroneFlybyPredictResponseDto:
    timings = {}
    t0 = time.perf_counter()

    if request.camera_command_feedback is not None:
        feedback = request.camera_command_feedback
        logger.warning('Camera command from frame %s was ignored: %s', feedback.frame, feedback.reason)

    annotations: List[DroneFlybyPredictionDto] = []
    requested_view = None
    debug = {'frame': request.frame, 'frame_index': request.frame_index}

    try:
        view = request.view
        region = view.source_region_xyxy
        level = view.resolution_level
        gap = 1 if state.last_frame_index is None else max(1, request.frame_index - state.last_frame_index)
        state.last_frame_index = request.frame_index
        state.frames += 1

        image = decode_view(view)
        timings['decode'] = time.perf_counter()

        try:
            state.ego.update(image, region, request.frame_index)
        except Exception:
            logger.exception('Ego-motion failed on frame %s (using the previous flow model)', request.frame)
        displacement = state.ego.displacement_at_center(gap)
        timings['ego'] = time.perf_counter()

        detector_error = None
        try:
            detections = get_detector().detect(image, request)
        except Exception as error:
            logger.exception('Detector failed on frame %s', request.frame)
            detections = []
            detector_error = f'{type(error).__name__}: {error}'
        timings['detect'] = time.perf_counter()
        _track_detector_health(state, request, detections, detector_error)

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

    return response


def _to_annotations(answers, request) -> List[DroneFlybyPredictionDto]:
    out = []
    for object_id, confidence, box in answers:
        # One bad answer (NaN confidence, unknown class...) must cost only itself,
        # not every other annotation in the frame.
        try:
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
        except Exception as error:
            logger.warning('Dropped one invalid annotation on frame %s (%s, confidence %r, box %s): %s',
                           request.frame, object_id, confidence, box, error)
    return out
