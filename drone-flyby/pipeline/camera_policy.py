"""Where to point the camera next.

``greedy`` (default) keeps a coarse grid over the source frame holding how much
detail we currently have about each cell (0 = never looked / long ago, 1 = just
seen at Level 2). The grid moves with the ground, so fresh ground entering the
frame shows up as zero-detail cells on the upstream edge, and it decays so old
looks lose value. Each frame, every legal (level, centre) the constraints allow
is scored by

    sum over covered cells of max(0, detail(level) - known detail)
  + bonus for uncertain tracks the view would see in more detail
  - small movement penalty

and the best one wins. Candidates are generated from ``camera_constraints``
and double-checked with ``utils.describe_camera_rejection``, so the command is
never refused.

``hold_l0`` and ``sweep_l1`` are simple reference policies for comparison.
"""

import logging
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from dtos import (
    FULL_FRAME_CENTER,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    SOURCE_REGION_SIZES,
    DroneFlybyPredictRequestDto,
    RequestedViewDto,
)
from pipeline import config
from pipeline.geometry import contained_fraction
from pipeline.tracker import Tracker
from utils import describe_camera_rejection, source_region_for_view

logger = logging.getLogger(__name__)

_SWEEP_L1 = [(960, 540), (1920, 540), (2880, 540), (2880, 1620), (1920, 1620), (960, 1620)]
# Every L1 waypoint is <=1102px (the L1 delta cap) from FULL_FRAME_CENTER, so a
# level-0 glance is always a legal single-frame detour from anywhere in the
# patrol. sweep_l1's 6-waypoint cycle is the minimum needed for full L1
# coverage under the per-frame movement cap (any 2 rows x 3 cols grid tiling
# the source at that cap needs all 6 stops), so a cell only gets a close look
# once per 6 frames -- something that enters and leaves inside that window is
# missed entirely regardless of waypoint order. A periodic low-res glance at
# the whole frame closes that gap without changing the L1 patrol's coverage.
# Interval is config.POLICY_SWEEP_GLANCE_EVERY (env-overridable, like every
# other policy knob).


class CameraPolicy:
    def __init__(self):
        self.cell = config.POLICY_CELL_PIXELS
        self.cols = math.ceil(IMAGE_WIDTH / self.cell)
        self.rows = math.ceil(IMAGE_HEIGHT / self.cell)
        self.detail = np.zeros((self.rows, self.cols), dtype=np.float32)
        self._carry = np.zeros(2)          # sub-cell ground motion not yet applied
        self._sweep_index = 0
        self._frames_since_glance = 0
        self.last_choice_debug = {}

    # ------------------------------------------------------------------ #
    # Memory of what we have seen
    # ------------------------------------------------------------------ #

    def advance(self, displacement: Tuple[float, float], gap: int) -> None:
        """Age the map and move it with the ground."""
        self.detail *= config.POLICY_DETAIL_DECAY ** gap
        self._carry += np.array(displacement, dtype=float)
        shift_cols = int(round(self._carry[0] / self.cell))
        shift_rows = int(round(self._carry[1] / self.cell))
        if shift_cols or shift_rows:
            self.detail = _shift_fill_zero(self.detail, shift_rows, shift_cols)
            self._carry -= np.array([shift_cols, shift_rows]) * self.cell

    def observe(self, region: Sequence[int], level: int) -> None:
        r0, r1, c0, c1 = self._cell_range(region)
        if r1 > r0 and c1 > c0:
            block = self.detail[r0:r1, c0:c1]
            np.maximum(block, config.POLICY_LEVEL_DETAIL[level], out=block)

    def _cell_range(self, region) -> Tuple[int, int, int, int]:
        c0 = int(np.clip(round(region[0] / self.cell), 0, self.cols))
        c1 = int(np.clip(round(region[2] / self.cell), 0, self.cols))
        r0 = int(np.clip(round(region[1] / self.cell), 0, self.rows))
        r1 = int(np.clip(round(region[3] / self.cell), 0, self.rows))
        return r0, r1, c0, c1

    # ------------------------------------------------------------------ #
    # Choosing
    # ------------------------------------------------------------------ #

    def choose(
        self,
        request: DroneFlybyPredictRequestDto,
        tracker: Tracker,
        velocity: np.ndarray,
        frame_index: int,
    ) -> Optional[RequestedViewDto]:
        mode = config.POLICY_MODE
        if mode == 'hold_l0':
            choice = (0, *FULL_FRAME_CENTER)
        elif mode == 'sweep_l1':
            choice = self._sweep(request)
        else:
            choice = self._greedy(request, tracker, velocity, frame_index)
        if choice is None:
            return None
        return self._legal_or_none(request, *choice)

    def _candidates(self, request: DroneFlybyPredictRequestDto) -> List[Tuple[int, int, int]]:
        constraints = request.camera_constraints
        current = request.view
        out = []
        for level in constraints.allowed_resolution_levels:
            if level == 0:
                out.append((0, *FULL_FRAME_CENTER))
                continue
            bounds = constraints.bounds_for_level(level)
            if bounds is None:
                continue
            xs = _grid(bounds.minimum_center_x, bounds.maximum_center_x, config.POLICY_CANDIDATE_STEP)
            ys = _grid(bounds.minimum_center_y, bounds.maximum_center_y, config.POLICY_CANDIDATE_STEP)
            # Holding still is always a candidate.
            if level == current.resolution_level:
                out.append((level, current.center_x, current.center_y))
            for x in xs:
                for y in ys:
                    if math.hypot(x - current.center_x, y - current.center_y) <= constraints.maximum_center_delta:
                        out.append((level, x, y))
        return out

    def _greedy(self, request, tracker: Tracker, velocity, frame_index):
        velocity = velocity if velocity is not None else np.zeros(2)
        gains = {
            level: np.clip(config.POLICY_LEVEL_DETAIL[level] - self.detail, 0.0, None)
            for level in (0, 1, 2)
        }
        integrals = {level: _integral(g) for level, g in gains.items()}
        current = request.view
        tracks = [(t.box, t.best_level, t.uncertainty(frame_index)) for t in tracker.tracks]

        best, best_score, scored = None, -1e18, 0
        for level, cx, cy in self._candidates(request):
            # The view is taken next frame, when the ground has moved by velocity:
            # we will see what is currently at region - velocity.
            region = source_region_for_view(level, cx, cy)
            looked_at = (
                region[0] - velocity[0], region[1] - velocity[1],
                region[2] - velocity[0], region[3] - velocity[1],
            )
            r0, r1, c0, c1 = self._cell_range(looked_at)
            coverage = _sum(integrals[level], r0, r1, c0, c1)

            bonus = 0.0
            scale = SOURCE_REGION_SIZES[level][0] / 960.0
            for box, best_level, uncertainty in tracks:
                if level < best_level or contained_fraction(box, looked_at) < 0.8:
                    continue
                longest_view = max(box[2] - box[0], box[3] - box[1]) / scale
                if longest_view < 6.0:
                    continue
                level_gain = 1.0 if level > best_level else 0.3
                bonus += level_gain * uncertainty

            distance = math.hypot(cx - current.center_x, cy - current.center_y) if level else 0.0
            score = (
                coverage + config.POLICY_TRACK_BONUS * bonus
            ) * config.POLICY_LEVEL_BIAS[level] - config.POLICY_MOVE_PENALTY * distance / 1000.0
            scored += 1
            if score > best_score:
                best, best_score = (level, cx, cy), score
                self.last_choice_debug = {
                    'level': level, 'center': (cx, cy), 'score': round(score, 2),
                    'coverage': round(coverage, 2), 'track_bonus': round(bonus, 2),
                }
        self.last_choice_debug['candidates'] = scored
        return best

    def _sweep(self, request):
        allowed = request.camera_constraints.allowed_resolution_levels
        if 1 not in allowed:
            # On L2: step back out to L1 first.
            return (1, *self._clamp_to_level(request, 1))

        self._frames_since_glance += 1
        if self._frames_since_glance > config.POLICY_SWEEP_GLANCE_EVERY and 0 in allowed:
            self._frames_since_glance = 0
            return (0, *FULL_FRAME_CENTER)

        target = _SWEEP_L1[self._sweep_index % len(_SWEEP_L1)]
        view = request.view
        if view.resolution_level == 1 and (view.center_x, view.center_y) == target:
            self._sweep_index += 1
            target = _SWEEP_L1[self._sweep_index % len(_SWEEP_L1)]
        return (1, *target)

    @staticmethod
    def _clamp_to_level(request, level):
        bounds = request.camera_constraints.bounds_for_level(level)
        x = min(max(request.view.center_x, bounds.minimum_center_x), bounds.maximum_center_x)
        y = min(max(request.view.center_y, bounds.minimum_center_y), bounds.maximum_center_y)
        return x, y

    @staticmethod
    def _legal_or_none(request, level, cx, cy) -> Optional[RequestedViewDto]:
        level, cx, cy = int(level), int(round(cx)), int(round(cy))
        view = request.view
        constraints = request.camera_constraints
        reason = describe_camera_rejection(
            view.resolution_level, (view.center_x, view.center_y), level, (cx, cy)
        )
        if reason is None and level not in constraints.allowed_resolution_levels:
            reason = f'level {level} not in allowed {constraints.allowed_resolution_levels}'
        if reason is None and level != 0 and math.hypot(cx - view.center_x, cy - view.center_y) > constraints.maximum_center_delta:
            reason = 'exceeds maximum_center_delta from the request'
        if reason is not None:
            logger.warning('Policy proposed an illegal view (%s, %s, %s): %s. Holding.', level, cx, cy, reason)
            return None
        if (level, cx, cy) == (view.resolution_level, view.center_x, view.center_y):
            return None  # holding is the same as sending nothing
        return RequestedViewDto(resolution_level=level, center_x=cx, center_y=cy)


# ---------------------------------------------------------------------- #

def _grid(lo: int, hi: int, step: int) -> List[int]:
    values = list(range(lo, hi + 1, step))
    if values[-1] != hi:
        values.append(hi)
    return values


def _integral(a: np.ndarray) -> np.ndarray:
    out = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.float64)
    out[1:, 1:] = a.cumsum(0).cumsum(1)
    return out


def _sum(integral: np.ndarray, r0, r1, c0, c1) -> float:
    if r1 <= r0 or c1 <= c0:
        return 0.0
    return float(integral[r1, c1] - integral[r0, c1] - integral[r1, c0] + integral[r0, c0])


def _shift_fill_zero(a: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Move content by (rows, cols); vacated cells are unexplored (0)."""
    out = np.zeros_like(a)
    h, w = a.shape
    if abs(rows) >= h or abs(cols) >= w:
        return out
    src_r = slice(max(0, -rows), h - max(0, rows))
    dst_r = slice(max(0, rows), h - max(0, -rows))
    src_c = slice(max(0, -cols), w - max(0, cols))
    dst_c = slice(max(0, cols), w - max(0, -cols))
    out[dst_r, dst_c] = a[src_r, src_c]
    return out
