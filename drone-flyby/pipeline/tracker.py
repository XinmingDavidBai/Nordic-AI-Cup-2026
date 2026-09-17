"""Frame-global object memory.

The response has to cover the whole source frame, but the camera only shows part
of it. The tracker keeps every object ever seen in source coordinates, moves it
with the estimated ground motion every frame, refines it whenever the camera
sees it again, and turns all of that into whole-frame annotations with
confidences that reflect how much we still trust each one.

All boxes are source pixels. ``frame_index`` (not ``frame``) is used as the
clock, since gaps in it are the frames we were too slow for.
"""

import itertools
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from dtos import IMAGE_HEIGHT, IMAGE_WIDTH
from pipeline import config
from pipeline.detector import Detection
from pipeline.egomotion import EgoMotionEstimator, FlowSample
from pipeline.geometry import (
    FRAME_BOX,
    Box,
    center,
    clip_to_frame,
    contained_fraction,
    iou,
    view_scale,
)

logger = logging.getLogger(__name__)

_ids = itertools.count(1)

# Detections within this many view pixels of a view edge that is not also a
# frame edge are probably truncated by the crop.
_EDGE_MARGIN_VIEW_PIXELS = 2.0
# A track this long (in view pixels) at the current level should be detectable,
# so not seeing it counts as evidence against it.
_MIN_DETECTABLE_VIEW_PIXELS = 8.0
# A detection this close to the frame border (at least this many source pixels,
# or _EDGE_MARGIN_VIEW_PIXELS view pixels if that is more) is cut by the frame.
_FRAME_EDGE_MARGIN_SOURCE_PIXELS = 4.0


def _frame_edges_touched(box, scale: float) -> List[bool]:
    """[left, top, right, bottom]: which frame borders this box is cut by."""
    margin = max(_FRAME_EDGE_MARGIN_SOURCE_PIXELS, _EDGE_MARGIN_VIEW_PIXELS * scale)
    return [
        box[0] <= margin,
        box[1] <= margin,
        box[2] >= IMAGE_WIDTH - margin,
        box[3] >= IMAGE_HEIGHT - margin,
    ]


def _in_frame_extent(box) -> float:
    """Shortest side of the part of the box inside the frame, in source px (0 if outside)."""
    return max(0.0, min(min(box[2], IMAGE_WIDTH) - max(box[0], 0.0), min(box[3], IMAGE_HEIGHT) - max(box[1], 0.0)))


@dataclass
class Track:
    box: np.ndarray                       # [x1, y1, x2, y2] source px
    created_index: int
    last_seen_index: int
    id: int = field(default_factory=lambda: next(_ids))
    position_weight: float = 0.0
    # Per axis (x, y): how much the current width/height is backed by complete observations.
    size_weight: np.ndarray = field(default_factory=lambda: np.zeros(2))
    # [left, top, right, bottom]: the object extends past this frame edge by an unknown amount.
    open_sides: List[bool] = field(default_factory=lambda: [False, False, False, False])
    confidence: float = 0.0               # weighted mean detector confidence
    confidence_weight: float = 0.0
    class_votes: Dict[str, float] = field(default_factory=dict)
    hits: int = 0
    misses: float = 0.0
    best_level: int = 0

    # -- derived ---------------------------------------------------------- #

    def ranked_classes(self) -> List[Tuple[str, float]]:
        total = sum(self.class_votes.values()) or 1.0
        return sorted(((k, v / total) for k, v in self.class_votes.items()), key=lambda kv: -kv[1])

    @property
    def object_id(self) -> str:
        return self.ranked_classes()[0][0]

    @property
    def purity(self) -> float:
        return self.ranked_classes()[0][1]

    def uncertainty(self, frame_index: int) -> float:
        """0 = we know exactly what and where this is, ~1+ = worth another look."""
        longest = max(self.box[2] - self.box[0], self.box[3] - self.box[1])
        # Small objects only seen from far away are the ones zoom helps most.
        detail_gap = {0: 1.0, 1: 0.4, 2: 0.0}[self.best_level] * min(1.0, 80.0 / max(longest, 1.0))
        staleness = min(1.0, (frame_index - self.last_seen_index) / max(config.TRACK_COAST_HALF_LIFE, 1.0))
        tentative = 0.5 if self.hits < config.TRACK_TENTATIVE_HITS else 0.0
        return (1.0 - self.purity) + detail_gap + 0.5 * staleness + tentative


class Tracker:
    def __init__(self):
        self.tracks: List[Track] = []
        # Ground-motion evidence from matched tracks, for EgoMotionEstimator.add_samples.
        self.flow_samples: List[FlowSample] = []

    # ------------------------------------------------------------------ #

    def step(
        self,
        frame_index: int,
        ego: EgoMotionEstimator,
        detections: Sequence[Detection],
        region: Sequence[int],
        level: int,
        gap: int,
    ) -> None:
        for track in self.tracks:
            track.box = ego.move_box(track.box, gap)
            self._pin_open_sides(track, ego)
            track.position_weight *= 0.5 ** gap

        # Partly visible objects stay in the answer: the ground truth keeps a
        # box for them down to a sliver, so only drop what has really left.
        self.tracks = [
            t for t in self.tracks
            if _in_frame_extent(t.box) >= config.TRACK_MIN_IN_FRAME_PIXELS
            and contained_fraction(t.box, FRAME_BOX) >= config.TRACK_MIN_IN_FRAME_FRACTION
            and frame_index - t.last_seen_index <= config.TRACK_MAX_COAST_FRAMES
        ]

        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        self.flow_samples = []
        for track_index, detection_index in matches:
            track = self.tracks[track_index]
            detection = detections[detection_index]
            since = max(1, frame_index - track.last_seen_index)
            if (
                since <= 10
                and not any(_frame_edges_touched(detection.box, view_scale(region)))
                and not any(track.open_sides)
                and track.size_weight.min() > 0  # otherwise the track centre is a cut box's centre
                and not self._touches_internal_view_edge(detection.box, region)
            ):
                # Where the flow model put it vs where it is: the per-frame
                # error, added to the model's own flow, is a flow measurement.
                det_c, trk_c = center(detection.box), center(track.box)
                predicted = ego.flow_at(*det_c)
                self.flow_samples.append(FlowSample(
                    x=det_c[0], y=det_c[1],
                    dx=float(predicted[0] + (det_c[0] - trk_c[0]) / since),
                    dy=float(predicted[1] + (det_c[1] - trk_c[1]) / since),
                    weight=0.2 * config.LEVEL_WEIGHT[level] * detection.confidence,
                    frame_index=frame_index,
                ))
            self._update(track, detection, region, level, frame_index)

        scale = view_scale(region)
        for track_index in unmatched_tracks:
            track = self.tracks[track_index]
            if contained_fraction(track.box, region) < 0.9:
                continue
            longest_view = max(track.box[2] - track.box[0], track.box[3] - track.box[1]) / scale
            if longest_view < _MIN_DETECTABLE_VIEW_PIXELS:
                continue
            # Missing it at a coarser level than we once saw it at is weak evidence.
            track.misses += 1.0 if level >= track.best_level else 0.5

        self.tracks = [
            t for t in self.tracks
            if t.misses < config.TRACK_MAX_MISSES
            and not (t.hits < config.TRACK_TENTATIVE_HITS and t.misses >= 1.0)
        ]

        for detection_index in unmatched_detections:
            detection = detections[detection_index]
            if detection.confidence < config.TRACK_NEW_MIN_CONF:
                continue
            track = Track(
                box=np.array(detection.box, dtype=float),
                created_index=frame_index,
                last_seen_index=frame_index,
            )
            self._update(track, detection, region, level, frame_index, new=True)
            self.tracks.append(track)

        self._merge_duplicates()

    # ------------------------------------------------------------------ #

    def _match(self, detections: Sequence[Detection]):
        pairs = []
        for ti, track in enumerate(self.tracks):
            tc = center(track.box)
            diagonal = float(np.hypot(track.box[2] - track.box[0], track.box[3] - track.box[1]))
            gate = max(config.TRACK_MATCH_MAX_DISTANCE, 0.75 * diagonal)
            for di, detection in enumerate(detections):
                overlap = iou(track.box, detection.box)
                dc = center(detection.box)
                distance = float(np.hypot(dc[0] - tc[0], dc[1] - tc[1]))
                if overlap < config.TRACK_MATCH_MIN_IOU and distance > gate:
                    continue
                pairs.append((overlap + 0.5 * max(0.0, 1.0 - distance / gate), ti, di))
        pairs.sort(reverse=True)

        used_tracks, used_detections, matches = set(), set(), []
        for _, ti, di in pairs:
            if ti in used_tracks or di in used_detections:
                continue
            used_tracks.add(ti)
            used_detections.add(di)
            matches.append((ti, di))
        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in used_tracks]
        unmatched_detections = [i for i in range(len(detections)) if i not in used_detections]
        return matches, unmatched_tracks, unmatched_detections

    def _update(self, track: Track, detection: Detection, region, level, frame_index, new=False):
        level_weight = config.LEVEL_WEIGHT[level]
        weight = level_weight * max(detection.confidence, 1e-3)
        position_w = size_w = weight

        det = np.array(detection.box, dtype=float)
        at_frame_edge = _frame_edges_touched(det, view_scale(region))
        at_view_edge = self._internal_view_edges_touched(det, region)
        cut = [f or v for f, v in zip(at_frame_edge, at_view_edge)]
        box = det.copy() if new else track.box.copy()
        for axis, (lo, hi) in enumerate(((0, 2), (1, 3))):
            if cut[lo] or cut[hi]:
                # Partly visible (cut by the frame or by the view): only the
                # inner side is real. Never average the cut size or centre in.
                box[lo], box[hi], open_lo, open_hi, known = self._edge_axis(track, det, axis, lo, hi, cut, new, position_w)
                # Only frame edges stay open (pinned, growing); a view edge
                # moves with the camera, so there the size is just unknown.
                track.open_sides[lo] = open_lo and at_frame_edge[lo]
                track.open_sides[hi] = open_hi and at_frame_edge[hi]
                if not known:
                    track.size_weight[axis] = 0.0
            elif new or track.open_sides[lo] or track.open_sides[hi] or track.size_weight[axis] <= 0:
                # First complete view of this axis: newest size wins outright.
                box[lo], box[hi] = det[lo], det[hi]
                track.open_sides[lo] = track.open_sides[hi] = False
                track.size_weight[axis] = size_w
            else:
                tc, dc = (box[lo] + box[hi]) / 2.0, (det[lo] + det[hi]) / 2.0
                ts, ds = box[hi] - box[lo], det[hi] - det[lo]
                c = (tc * track.position_weight + dc * position_w) / (track.position_weight + position_w)
                s = (ts * track.size_weight[axis] + ds * size_w) / (track.size_weight[axis] + size_w)
                box[lo], box[hi] = c - s / 2.0, c + s / 2.0
                track.size_weight[axis] = min(track.size_weight[axis] + size_w, 5.0)
        track.box = box
        track.position_weight = position_w if new else track.position_weight + position_w

        track.confidence = (
            track.confidence * track.confidence_weight + detection.confidence * level_weight
        ) / (track.confidence_weight + level_weight)
        track.confidence_weight = min(track.confidence_weight + level_weight, 5.0)
        track.class_votes[detection.object_id] = (
            track.class_votes.get(detection.object_id, 0.0) + weight
        )
        track.hits += 1
        track.misses = 0.0
        track.last_seen_index = frame_index
        track.best_level = max(track.best_level, level)

    @staticmethod
    def _edge_axis(track: Track, det, axis, lo, hi, at_edge, new, weight):
        """Box sides along one axis for a detection that touches the frame edge.

        Returns (side_lo, side_hi, open_lo, open_hi, full_size_known). An open
        side means "the object continues beyond the frame by an unknown amount".
        If the track was seen complete along this axis before (an object on its
        way out), that size is kept and the box simply extends past the edge.
        The uncut (inner) side is a real measurement, so when the track's own
        inner side is real too, the two are averaged like any other position.
        """
        if at_edge[lo] and at_edge[hi]:
            return det[lo], det[hi], True, True, False
        known_size = None
        if not new and track.size_weight[axis] > 0 and not (track.open_sides[lo] or track.open_sides[hi]):
            known_size = track.box[hi] - track.box[lo]

        def inner_side(side, cut_side):
            # Complete track, or one still entering across this same edge.
            real = known_size is not None or (track.open_sides[cut_side] and not track.open_sides[side])
            if new or not real:
                return det[side]
            return (track.box[side] * track.position_weight + det[side] * weight) / (track.position_weight + weight)

        if at_edge[lo]:
            inner = inner_side(hi, lo)
            if known_size is not None and inner - known_size <= det[lo]:
                return inner - known_size, inner, False, False, True
            return det[lo], inner, True, False, False
        inner = inner_side(lo, hi)
        if known_size is not None and inner + known_size >= det[hi]:
            return inner, inner + known_size, False, False, True
        return inner, det[hi], False, True, False

    @staticmethod
    def _pin_open_sides(track: Track, ego: EgoMotionEstimator) -> None:
        """Let a box that is still entering the frame grow instead of sliding in as a sliver.

        An open side whose frame edge the ground is flowing in from stays pinned
        to that edge while the rest of the box moves with the flow, so the box
        grows with the part of the object that has come into view. Growth is
        capped (relative to the other, known dimension) so an object that
        entered completely while nobody was looking does not grow forever.
        """
        if not any(track.open_sides):
            return
        x1, y1, x2, y2 = (float(c) for c in track.box)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        left, top, right, bottom = track.open_sides
        if left and ego.flow_at(0.0, cy)[0] > 0:
            x1 = min(x1, 0.0)
        if right and ego.flow_at(IMAGE_WIDTH, cy)[0] < 0:
            x2 = max(x2, float(IMAGE_WIDTH))
        if top and ego.flow_at(cx, 0.0)[1] > 0:
            y1 = min(y1, 0.0)
        if bottom and ego.flow_at(cx, IMAGE_HEIGHT)[1] < 0:
            y2 = max(y2, float(IMAGE_HEIGHT))

        def cap(other_extent, other_open):
            if other_open:
                return config.TRACK_OPEN_MAX_EXTENT
            return max(config.TRACK_OPEN_MAX_ASPECT * other_extent, 1.0)

        if left or right:
            limit = cap(y2 - y1, top or bottom)
            if x2 - x1 > limit:
                if left:
                    x1 = x2 - limit
                else:
                    x2 = x1 + limit
                track.open_sides[0] = track.open_sides[2] = False
        if top or bottom:
            limit = cap(x2 - x1, left or right)
            if y2 - y1 > limit:
                if top:
                    y1 = y2 - limit
                else:
                    y2 = y1 + limit
                track.open_sides[1] = track.open_sides[3] = False
        track.box = np.array([x1, y1, x2, y2])

    @staticmethod
    def _internal_view_edges_touched(box, region) -> List[bool]:
        """[left, top, right, bottom]: view borders (that are not frame borders) cutting this box."""
        margin = _EDGE_MARGIN_VIEW_PIXELS * view_scale(region)
        return [
            region[0] > 0 and box[0] - region[0] <= margin,
            region[1] > 0 and box[1] - region[1] <= margin,
            region[2] < IMAGE_WIDTH and region[2] - box[2] <= margin,
            region[3] < IMAGE_HEIGHT and region[3] - box[3] <= margin,
        ]

    @staticmethod
    def _touches_internal_view_edge(box, region) -> bool:
        margin = _EDGE_MARGIN_VIEW_PIXELS * view_scale(region)
        return (
            (region[0] > 0 and box[0] - region[0] <= margin)
            or (region[1] > 0 and box[1] - region[1] <= margin)
            or (region[2] < IMAGE_WIDTH and region[2] - box[2] <= margin)
            or (region[3] < IMAGE_HEIGHT and region[3] - box[3] <= margin)
        )

    def _merge_duplicates(self) -> None:
        self.tracks.sort(key=lambda t: (-t.hits, t.created_index))
        kept: List[Track] = []
        for track in self.tracks:
            duplicate_of = next(
                (k for k in kept if iou(k.box, track.box) >= config.TRACK_DUPLICATE_IOU), None
            )
            if duplicate_of is None:
                kept.append(track)
                continue
            for name, votes in track.class_votes.items():
                duplicate_of.class_votes[name] = duplicate_of.class_votes.get(name, 0.0) + votes
            duplicate_of.hits += track.hits
            duplicate_of.last_seen_index = max(duplicate_of.last_seen_index, track.last_seen_index)
            duplicate_of.best_level = max(duplicate_of.best_level, track.best_level)
        self.tracks = kept

    # ------------------------------------------------------------------ #

    def outputs(self, frame_index: int) -> List[Tuple[str, float, Box]]:
        """Whole-frame answers: (object_id, confidence, source box)."""
        answers = []
        for track in self.tracks:
            box = clip_to_frame(track.box)
            if box[2] - box[0] < 1.0 or box[3] - box[1] < 1.0:
                continue
            age = frame_index - track.last_seen_index
            score = (
                track.confidence
                * min(1.0, 0.5 + 0.25 * track.hits)
                * 0.5 ** (age / max(config.TRACK_COAST_HALF_LIFE, 1e-3))
                * 0.6 ** track.misses
            )
            ranked = track.ranked_classes()
            best_name, best_share = ranked[0]
            main_score = score * (0.5 + 0.5 * best_share)
            if main_score >= config.TRACK_OUTPUT_MIN_SCORE:
                answers.append((best_name, float(min(main_score, 1.0)), box))
            # Hedging on the runner-up class is cheap under per-class AP: a wrong
            # low-confidence guess ranks last in that class.
            if len(ranked) > 1 and config.TRACK_RUNNER_UP_MIN_SHARE > 0:
                name, share = ranked[1]
                runner_score = score * share * 0.5
                if share >= config.TRACK_RUNNER_UP_MIN_SHARE and runner_score >= config.TRACK_OUTPUT_MIN_SCORE:
                    answers.append((name, float(min(runner_score, 1.0)), box))
        answers.sort(key=lambda a: -a[1])
        return answers[:500]
