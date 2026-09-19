"""Detector backends. Every backend returns detections in SOURCE pixels.

Pick one with ``DETECTOR_BACKEND`` (see config.py). All of them take the decoded
960x540 BGR view plus the request, because lifting a box out of the view needs
``source_region_xyxy``.
"""

import hashlib
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from dtos import OBJECT_CLASSES, DroneFlybyPredictRequestDto
from pipeline import config
from pipeline.device import DeviceChoice, select_device
from pipeline.geometry import (
    Box,
    area,
    contained_fraction,
    intersection,
    view_pixels_to_source,
    view_scale,
)

logger = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def log_device(choice: DeviceChoice) -> None:
    """One clear line about where the detector runs; an error when a GPU goes unused."""
    if choice.gpu_hardware_unused and choice.how != 'forced':
        logger.error('!!! DETECTOR ON CPU although GPU hardware is present: %s (torch %s, %s). '
                     'Fix the torch build / driver / container (--gpus all) for the latency budget.',
                     choice.summary(), choice.torch_version, choice.torch_build)
    elif choice.kind == 'cpu' and choice.how != 'forced':
        logger.warning('Detector on CPU: %s (torch %s, %s)', choice.summary(), choice.torch_version, choice.torch_build)
    else:
        logger.info('Detector %s (torch %s, %s)', choice.summary(), choice.torch_version, choice.torch_build)


@dataclass
class Detection:
    object_id: str
    confidence: float
    box: Box  # source pixels [x1, y1, x2, y2]


class Detector:
    name = 'base'
    # Set by warmup() when the dummy inference raised; api.py refuses to start then.
    warmup_error: Optional[str] = None

    def detect(self, image: np.ndarray, request: DroneFlybyPredictRequestDto) -> List[Detection]:
        raise NotImplementedError

    def warmup(self) -> None:
        dummy = np.zeros((540, 960, 3), dtype=np.uint8)
        try:
            self._warmup_image(dummy)
            self.warmup_error = None
        except Exception as error:
            self.warmup_error = f'{type(error).__name__}: {error}'
            logger.exception('Detector warmup failed (continuing)')

    def _warmup_image(self, image: np.ndarray) -> None:
        pass

    def describe(self) -> dict:
        """What is loaded, for the /api endpoint and the startup check."""
        return {'backend': self.name, 'warmup_error': self.warmup_error}


class NullDetector(Detector):
    name = 'none'

    def detect(self, image, request):
        return []


class YoloDetector(Detector):
    """Ultralytics YOLO trained by ``train_detector.py``.

    The training script writes class ids in ``dtos.OBJECT_CLASSES`` order, but
    names are mapped through ``model.names`` anyway so a model trained with a
    different order still works, and unknown names are dropped rather than
    failing the response.
    """

    name = 'yolo'

    def __init__(self, weights_path):
        import ultralytics
        from ultralytics import YOLO

        weights_path = Path(weights_path)
        # Without this, ultralytics treats a missing path as a name to download.
        if not weights_path.is_file():
            raise FileNotFoundError(f'DETECTOR_WEIGHTS {weights_path} does not exist')
        self.weights_path = weights_path.resolve()
        self.weights_sha256 = _sha256(self.weights_path)
        self.weights_bytes = self.weights_path.stat().st_size
        self.ultralytics_version = ultralytics.__version__
        self.model = YOLO(str(weights_path))
        # Raises for a forced device torch cannot use (api.py then refuses to start).
        self.device_choice = select_device(config.DETECTOR_DEVICE)
        self.device = self.device_choice.device
        self.names = {int(k): v for k, v in self.model.names.items()}
        self.gamma_lut = None
        if config.DETECTOR_GAMMA != 1.0:
            self.gamma_lut = np.clip(255.0 * (np.arange(256) / 255.0) ** config.DETECTOR_GAMMA + 0.5, 0, 255).astype(np.uint8)
            logger.info('Detector input gamma %.2f (DETECTOR_GAMMA)', config.DETECTOR_GAMMA)
        unknown = [n for n in self.names.values() if n not in OBJECT_CLASSES]
        if unknown:
            logger.warning('Model has classes the protocol does not accept, they will be dropped: %s', unknown)
        logger.info(
            'YOLO detector loaded from %s (sha256 %s)', self.weights_path, self.weights_sha256[:12],
        )
        log_device(self.device_choice)

    def describe(self):
        return {
            **super().describe(),
            'weights_path': str(self.weights_path),
            'weights_sha256': self.weights_sha256[:12],
            'weights_bytes': self.weights_bytes,
            'num_classes': len(self.names),
            'device': self.device,
            'device_info': self.device_choice.as_dict(),
            'ultralytics_version': self.ultralytics_version,
            'input_gamma': config.DETECTOR_GAMMA,
        }

    def _run(self, image: np.ndarray):
        if self.gamma_lut is not None:
            image = self.gamma_lut[image]
        return self.model.predict(
            image,  # BGR numpy, which is what ultralytics expects for arrays
            imgsz=config.DETECTOR_IMGSZ,
            conf=config.DETECTOR_MIN_CONF,
            iou=config.DETECTOR_NMS_IOU,
            agnostic_nms=True,
            half=config.DETECTOR_HALF and self.device_choice.kind in ('cuda', 'rocm'),
            device=self.device,
            verbose=False,
        )[0]

    def _warmup_image(self, image):
        for _ in range(2):
            self._run(image)

    def warmup(self) -> None:
        super().warmup()
        if self.warmup_error is None or not self.device_choice.is_gpu or self.device_choice.how != 'auto':
            return
        # Auto-picked GPU that cannot actually run the model (driver/kernel mismatch,
        # out of memory...): serving on the CPU beats serving nothing. A forced GPU
        # stays failed, so api.py refuses to start instead.
        failed = self.device_choice
        logger.error('!!! GPU %s (%s) failed the warmup inference: %s. FALLING BACK TO CPU; '
                     'expect several times the latency.', failed.device, failed.gpu_name, self.warmup_error)
        self.device_choice = DeviceChoice(
            'cpu', 'cpu', 'fallback',
            f'{failed.kind.upper()} GPU {failed.device} ({failed.gpu_name}) failed warmup: {self.warmup_error}',
            requested=failed.requested, torch_version=failed.torch_version, torch_build=failed.torch_build,
            gpu_hardware_seen=failed.gpu_hardware_seen, gpu_hardware_unused=True,
        )
        self.device = 'cpu'
        self.model.predictor = None   # rebuilt on the next call, on the new device
        super().warmup()
        log_device(self.device_choice)

    def detect(self, image, request):
        result = self._run(image)
        region = request.view.source_region_xyxy
        detections: List[Detection] = []
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        for box, score, cls in zip(boxes, scores, classes):
            name = self.names.get(int(cls))
            if name not in OBJECT_CLASSES:
                continue
            detections.append(
                Detection(name, float(score), view_pixels_to_source(box.tolist(), region))
            )
        return detections


class GroundTruthDetector(Detector):
    """LOCAL DEBUG ONLY: 'detects' the supplied annotations the camera can see.

    Use it to measure what tracking + camera policy can achieve with a perfect
    detector (an upper bound), and to debug them without a trained model. An
    object counts as visible if enough of it is inside the view and it is at
    least GT_MIN_VIEW_PIXELS long in transmitted pixels, so zooming matters.
    """

    name = 'gt'

    def __init__(self):
        from utils import DEFAULT_SCENE

        self.scene = DEFAULT_SCENE
        self._cache = {}
        self._rng = random.Random(0)

    def detect(self, image, request):
        if request.sequence_id != 'local':
            raise RuntimeError('The gt detector only runs against local_evaluator.py')
        from utils import load_annotations

        if request.frame not in self._cache:
            self._cache[request.frame] = load_annotations(request.frame, self.scene)
        region = request.view.source_region_xyxy
        scale = view_scale(region)
        noise = config.GT_BOX_NOISE_PIXELS * scale
        detections = []
        for annotation in self._cache[request.frame]:
            box = tuple(float(c) for c in annotation['bbox'])
            if contained_fraction(box, region) < config.GT_MIN_VISIBLE_FRACTION:
                continue
            visible = intersection(box, region)
            if area(visible) <= 0:
                continue
            longest = max(visible[2] - visible[0], visible[3] - visible[1]) / scale
            shortest = min(visible[2] - visible[0], visible[3] - visible[1]) / scale
            # Partly visible objects (cut by the view or frame edge) are reported
            # as their visible part, like a real detector would, as long as that
            # part is big enough to see at all.
            if longest < config.GT_MIN_VIEW_PIXELS or shortest < config.GT_MIN_PARTIAL_VIEW_PIXELS:
                continue
            jittered = tuple(c + self._rng.gauss(0.0, noise) for c in visible)
            # Small objects at coarse levels get lower confidence, like a real model.
            confidence = min(0.99, 0.4 + 0.03 * longest)
            detections.append(Detection(annotation['object_id'], confidence, jittered))
        return detections


class EdgeBaselineDetector(Detector):
    """The organisers' example.detect, adapted to return source-pixel boxes."""

    name = 'edges'

    def detect(self, image, request):
        from example import detect as baseline_detect
        from utils import global_bbox_to_source

        out = []
        for annotation in baseline_detect(image, request):
            box = global_bbox_to_source(annotation.bbox, request.original_width, request.original_height)
            out.append(Detection(annotation.object_id, float(annotation.confidence), box))
        return out


def build_detector() -> Detector:
    backend = config.DETECTOR_BACKEND.lower()
    if backend == 'auto':
        if config.DETECTOR_WEIGHTS.is_file():
            backend = 'yolo'
        else:
            logger.warning(
                'No detector weights at %s, so nothing will be detected. Train one with '
                '`python train_detector.py`, or set DETECTOR_BACKEND=gt to debug the rest '
                'of the pipeline against local ground truth.',
                config.DETECTOR_WEIGHTS,
            )
            backend = 'none'
    if backend == 'yolo':
        return YoloDetector(config.DETECTOR_WEIGHTS)
    if backend == 'gt':
        logger.warning('Using the GROUND-TRUTH detector. Local debugging only, never deploy this.')
        return GroundTruthDetector()
    if backend == 'edges':
        return EdgeBaselineDetector()
    if backend == 'none':
        return NullDetector()
    raise ValueError(f'Unknown DETECTOR_BACKEND {config.DETECTOR_BACKEND!r}')
