"""Model how the ground moves across the source frame from one frame to the next.

The drone flies a straight line, but the camera is not looking straight down,
so the motion is not a single shift. On helsinki (fitted from the annotations)
objects move ~54 px/frame at the top of the frame and ~80 px/frame at the
bottom, and spread outward horizontally: a translation-only model is off by
several pixels per frame, which is enough to push coasting boxes of 20-40 px
objects below IoU 0.5 within a few frames.

So the model is an affine flow field, per frame, in source pixels:

    dx(x, y) = a0 + a1 * X + a2 * Y        X = (x - 1920) / 1000
    dy(x, y) = b0 + b1 * X + b2 * Y        Y = (y - 1080) / 1000

fitted by weighted ridge regression (towards a prior) over recent samples.

Samples come from two places:

* image registration: the part of the source frame both the previous and the
  current view covered is rendered at a common scale, cut into tiles, and each
  tile is phase-correlated. That yields a local displacement at the tile's
  location. Camera moves cancel out because everything is in source
  coordinates. (Checked on helsinki: ``cv2.phaseCorrelate(previous, current)``
  is +dy when content moves down.)
* matched tracks (see Tracker.flow_samples), which keep the model alive when
  consecutive views do not overlap.
"""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from dtos import FULL_FRAME_CENTER
from pipeline import config
from pipeline.geometry import view_scale

logger = logging.getLogger(__name__)

# Flow parameters fitted on helsinki's annotations (rows: const, X, Y; cols: dx, dy).
HELSINKI_PRIOR = np.array([
    [-0.09, 65.62],
    [6.69, 0.08],
    [-0.03, 11.36],
])


@dataclass
class _View:
    gray: np.ndarray
    region: Tuple[int, int, int, int]
    frame_index: int


@dataclass
class FlowSample:
    x: float
    y: float
    dx: float          # per frame
    dy: float          # per frame
    weight: float
    frame_index: int


def _features(x, y) -> np.ndarray:
    return np.array([1.0, (x - FULL_FRAME_CENTER[0]) / 1000.0, (y - FULL_FRAME_CENTER[1]) / 1000.0])


class EgoMotionEstimator:
    def __init__(self):
        self.previous: Optional[_View] = None
        self.samples: Deque[FlowSample] = deque(maxlen=2000)
        prior = HELSINKI_PRIOR if config.EGO_PRIOR == 'helsinki' else np.zeros((3, 2))
        self.prior = prior.copy()
        self.theta = prior.copy()          # (3, 2)
        self.has_measurement = False
        self.last_measurement_ok = False
        self.last_response = 0.0
        self.last_tile_count = 0
        self._bad_frames = 0

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def flow_at(self, x: float, y: float) -> np.ndarray:
        """Per-frame ground displacement (dx, dy) at source location (x, y)."""
        return _features(x, y) @ self.theta

    @property
    def velocity(self) -> np.ndarray:
        """Flow at the frame centre, per frame."""
        return self.flow_at(*FULL_FRAME_CENTER)

    def velocity_or_zero(self) -> np.ndarray:
        return self.velocity

    def move_box(self, box: Sequence[float], frames: int) -> np.ndarray:
        """Carry a source box forward ``frames`` frames along the flow field."""
        x1, y1, x2, y2 = (float(c) for c in box)
        for _ in range(max(0, frames)):
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            x1 += self.flow_at(x1, cy)[0]
            x2 += self.flow_at(x2, cy)[0]
            y1 += self.flow_at(cx, y1)[1]
            y2 += self.flow_at(cx, y2)[1]
        return np.array([x1, y1, x2, y2])

    def displacement_at_center(self, frames: int) -> Tuple[float, float]:
        moved = self.move_box((FULL_FRAME_CENTER[0], FULL_FRAME_CENTER[1]) * 2, frames)
        return float(moved[0] - FULL_FRAME_CENTER[0]), float(moved[1] - FULL_FRAME_CENTER[1])

    # ------------------------------------------------------------------ #
    # Updates
    # ------------------------------------------------------------------ #

    def update(self, image_bgr: np.ndarray, region: Sequence[int], frame_index: int) -> bool:
        """Register the new view against the previous one and refit. Returns success."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        current = _View(gray, tuple(int(c) for c in region), frame_index)
        previous, self.previous = self.previous, current
        self.last_measurement_ok = False
        self.last_tile_count = 0
        self.last_response = 0.0
        if previous is None:
            return False

        gap = max(1, frame_index - previous.frame_index)
        measured = self._measure_tiles(previous, current, gap)
        accepted = []
        for sample in measured:
            predicted = self.flow_at(sample.x, sample.y)
            jump = np.hypot(sample.dx - predicted[0], sample.dy - predicted[1])
            if np.hypot(sample.dx, sample.dy) > config.EGO_MAX_SPEED:
                continue
            if self.has_measurement and jump > config.EGO_MAX_VELOCITY_JUMP:
                continue
            accepted.append(sample)

        if measured and not accepted:
            self._bad_frames += 1
            if self._bad_frames >= 3:
                # Three frames of disagreement in a row: the model is what's wrong.
                logger.warning('Ego-motion model disagreed with 3 frames of registration; refitting from scratch')
                self.samples.clear()
                accepted = [s for s in measured if np.hypot(s.dx, s.dy) <= config.EGO_MAX_SPEED]
                self._bad_frames = 0
        elif accepted:
            self._bad_frames = 0

        if accepted:
            self.samples.extend(accepted)
            self.has_measurement = True
            self.last_measurement_ok = True
            self.last_tile_count = len(accepted)
            self._fit(frame_index)
        return self.last_measurement_ok

    def add_samples(self, samples: List[FlowSample], frame_index: int) -> None:
        if not samples:
            return
        self.samples.extend(samples)
        self.has_measurement = True
        self._fit(frame_index)

    def _fit(self, frame_index: int) -> None:
        live = [s for s in self.samples if frame_index - s.frame_index <= config.EGO_SAMPLE_WINDOW]
        if not live:
            return
        features = np.array([_features(s.x, s.y) for s in live])
        targets = np.array([[s.dx, s.dy] for s in live])
        weights = np.array([
            s.weight * config.EGO_SAMPLE_DECAY ** (frame_index - s.frame_index) for s in live
        ])
        # The constant term is weakly regularised; the gradient terms (hard to
        # observe from a single small view) lean on the prior more.
        ridge = np.diag([config.EGO_PRIOR_STRENGTH * 0.1, config.EGO_PRIOR_STRENGTH, config.EGO_PRIOR_STRENGTH])

        theta = self._solve(features, targets, weights, ridge)
        # One round of outlier rejection.
        residual = np.hypot(*(targets - features @ theta).T)
        limit = 3.0 * max(3.0, float(np.sqrt(np.average(residual ** 2, weights=weights))))
        keep = residual <= limit
        if keep.sum() >= 3 and not keep.all():
            theta = self._solve(features[keep], targets[keep], weights[keep], ridge)
        self.theta = theta

    def _solve(self, features, targets, weights, ridge) -> np.ndarray:
        weighted = features * weights[:, None]
        lhs = features.T @ weighted + ridge
        rhs = weighted.T @ targets + ridge @ self.prior
        return np.linalg.solve(lhs, rhs)

    # ------------------------------------------------------------------ #

    def _measure_tiles(self, previous: _View, current: _View, gap: int) -> List[FlowSample]:
        p, c = previous.region, current.region
        ix1, iy1 = max(p[0], c[0]), max(p[1], c[1])
        ix2, iy2 = min(p[2], c[2]), min(p[3], c[3])
        scale = max(view_scale(p), view_scale(c))
        out_w = int((ix2 - ix1) / scale)
        out_h = int((iy2 - iy1) / scale)
        if out_w < config.EGO_MIN_OVERLAP_PIXELS or out_h < config.EGO_MIN_OVERLAP_PIXELS:
            return []

        patches = []
        for view in (previous, current):
            s = view_scale(view.region)
            x1 = int(round((ix1 - view.region[0]) / s))
            y1 = int(round((iy1 - view.region[1]) / s))
            x2 = int(round((ix2 - view.region[0]) / s))
            y2 = int(round((iy2 - view.region[1]) / s))
            crop = view.gray[y1:y2, x1:x2]
            if crop.size == 0:
                return []
            if crop.shape[1] != out_w or crop.shape[0] != out_h:
                crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)
            patches.append(crop.astype(np.float32))

        # Tiles must be comfortably larger than the expected shift, or phase
        # correlation aliases. Expected shift is in common-scale pixels.
        expected = np.abs(self.velocity) * gap / scale
        min_tile_w = max(config.EGO_MIN_OVERLAP_PIXELS, 4.0 * expected[0])
        min_tile_h = max(config.EGO_MIN_OVERLAP_PIXELS, 4.0 * expected[1])
        nx = int(max(1, min(3, out_w // min_tile_w)))
        ny = int(max(1, min(3, out_h // min_tile_h)))

        samples, responses = [], []
        tile_w, tile_h = out_w // nx, out_h // ny
        window = cv2.createHanningWindow((tile_w, tile_h), cv2.CV_32F)
        for j in range(ny):
            for i in range(nx):
                a = patches[0][j * tile_h:(j + 1) * tile_h, i * tile_w:(i + 1) * tile_w]
                b = patches[1][j * tile_h:(j + 1) * tile_h, i * tile_w:(i + 1) * tile_w]
                if a.shape != (tile_h, tile_w) or b.shape != (tile_h, tile_w):
                    continue
                (dx, dy), response = cv2.phaseCorrelate(a, b, window)
                responses.append(response)
                if response < config.EGO_MIN_RESPONSE:
                    continue
                # Location: tile centre, halfway along the motion.
                x = ix1 + (i + 0.5) * tile_w * scale - dx * scale / 2.0
                y = iy1 + (j + 0.5) * tile_h * scale - dy * scale / 2.0
                samples.append(
                    FlowSample(x, y, dx * scale / gap, dy * scale / gap, float(response), current.frame_index)
                )
        self.last_response = float(np.mean(responses)) if responses else 0.0
        return samples
