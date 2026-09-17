"""Small box helpers. Every box in the pipeline is [x1, y1, x2, y2] in source pixels."""

from typing import Sequence, Tuple

import numpy as np

from dtos import IMAGE_HEIGHT, IMAGE_WIDTH, TRANSMITTED_VIEW_SIZE

Box = Tuple[float, float, float, float]


def area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection(a: Sequence[float], b: Sequence[float]) -> Box:
    return (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    inter = area(intersection(a, b))
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def contained_fraction(box: Sequence[float], region: Sequence[float]) -> float:
    """Share of ``box`` that lies inside ``region``."""
    box_area = area(box)
    return area(intersection(box, region)) / box_area if box_area > 0 else 0.0


def center(box: Sequence[float]) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def shift(box: Sequence[float], dx: float, dy: float) -> Box:
    return (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)


def clip_to_frame(box: Sequence[float]) -> Box:
    return (
        min(max(box[0], 0.0), IMAGE_WIDTH),
        min(max(box[1], 0.0), IMAGE_HEIGHT),
        min(max(box[2], 0.0), IMAGE_WIDTH),
        min(max(box[3], 0.0), IMAGE_HEIGHT),
    )


FRAME_BOX: Box = (0.0, 0.0, float(IMAGE_WIDTH), float(IMAGE_HEIGHT))


def view_scale(source_region: Sequence[int]) -> float:
    """Source pixels per transmitted-view pixel (4, 2 or 1)."""
    return (source_region[2] - source_region[0]) / float(TRANSMITTED_VIEW_SIZE[0])


def view_pixels_to_source(box: Sequence[float], source_region: Sequence[int]) -> Box:
    """Map a box in transmitted-view pixels (960x540) into source pixels."""
    sx = (source_region[2] - source_region[0]) / float(TRANSMITTED_VIEW_SIZE[0])
    sy = (source_region[3] - source_region[1]) / float(TRANSMITTED_VIEW_SIZE[1])
    return (
        source_region[0] + box[0] * sx,
        source_region[1] + box[1] * sy,
        source_region[0] + box[2] * sx,
        source_region[1] + box[3] * sy,
    )


def source_to_view_pixels(box: Sequence[float], source_region: Sequence[int]) -> Box:
    sx = (source_region[2] - source_region[0]) / float(TRANSMITTED_VIEW_SIZE[0])
    sy = (source_region[3] - source_region[1]) / float(TRANSMITTED_VIEW_SIZE[1])
    return (
        (box[0] - source_region[0]) / sx,
        (box[1] - source_region[1]) / sy,
        (box[2] - source_region[0]) / sx,
        (box[3] - source_region[1]) / sy,
    )


def nms_order(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float):
    """Class-agnostic NMS. Returns kept indices, highest score first."""
    order = list(np.argsort(-scores))
    keep = []
    while order:
        index = order.pop(0)
        keep.append(index)
        order = [j for j in order if iou(boxes[index], boxes[j]) < iou_threshold]
    return keep
