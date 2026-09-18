"""Every tunable knob of the pipeline in one place.

Each value can be overridden with an environment variable of the same name
(``DETECTOR_BACKEND=gt python api.py``), or by putting ``NAME=value`` lines in
a ``.env`` file next to ``api.py``. ``.env`` is gitignored.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader: NAME=value per line, # comments, no overrides."""
    if not path.is_file():
        return
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        name, value = line.split('=', 1)
        os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT / '.env')


def _str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ('1', 'true', 'yes', 'on')


# =========================================================================== #
#                                  API KEY                                    #
# =========================================================================== #
#
#   >>> YOUR NORDIC AI CUP TEAM API KEY GOES HERE <<<
#
#   Preferred: copy `.env.example` to `.env` and fill in NAIC_API_KEY there
#   (.env is gitignored, so the key never lands in a commit).
#   Alternative: replace the empty string below. Do NOT commit it if you do.
#
#   Nothing in this pipeline sends the key anywhere and nothing submits
#   anything. The evaluation service calls *your* endpoint; the key is what you
#   paste into the form at https://cases.nordicaicup.com together with your
#   endpoint URL when you choose to verify / validate / submit.
#
NAIC_API_KEY = _str('NAIC_API_KEY', '')  # <<< INSERT API KEY HERE (or in .env)
#
# =========================================================================== #


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #

# auto  : yolo if DETECTOR_WEIGHTS exists, otherwise 'none' (with a warning)
# yolo  : ultralytics model at DETECTOR_WEIGHTS
# gt    : LOCAL DEBUG ONLY. Reads the supplied ground truth for the frame and
#         reports what the camera could plausibly see. Lets you debug tracking
#         and the camera policy independently of the detector. Refuses to run
#         on any sequence that is not the local evaluator's.
# edges : the organisers' edge-detection baseline (plumbing only)
# none  : no detections
DETECTOR_BACKEND = _str('DETECTOR_BACKEND', 'auto')
DETECTOR_WEIGHTS = Path(_str('DETECTOR_WEIGHTS', str(ROOT / 'weights' / 'detector.pt')))
DETECTOR_DEVICE = _str('DETECTOR_DEVICE', '')          # '' = auto (GPU 0 if torch sees one: CUDA or ROCm)
DETECTOR_IMGSZ = _int('DETECTOR_IMGSZ', 960)           # try 1280 for tiny objects
DETECTOR_MIN_CONF = _float('DETECTOR_MIN_CONF', 0.10)
DETECTOR_NMS_IOU = _float('DETECTOR_NMS_IOU', 0.50)
DETECTOR_HALF = _bool('DETECTOR_HALF', False)          # fp16, GPU only

# gt backend: minimum longest side, in transmitted-view pixels, for an object to
# count as "visible" at a level; optional minimum shortest side of the visible
# part (raise it to stop reporting thin slivers); and an optional minimum share
# of the box inside the view (0 = report partly visible objects; the old
# behaviour was 0.5).
GT_MIN_VIEW_PIXELS = _float('GT_MIN_VIEW_PIXELS', 10.0)
GT_MIN_PARTIAL_VIEW_PIXELS = _float('GT_MIN_PARTIAL_VIEW_PIXELS', 0.0)
GT_MIN_VISIBLE_FRACTION = _float('GT_MIN_VISIBLE_FRACTION', 0.0)
GT_BOX_NOISE_PIXELS = _float('GT_BOX_NOISE_PIXELS', 1.0)   # view pixels, 1 sigma

# --------------------------------------------------------------------------- #
# Ego-motion (how far the ground moved between two frames)
# --------------------------------------------------------------------------- #

EGO_MIN_OVERLAP_PIXELS = _int('EGO_MIN_OVERLAP_PIXELS', 96)   # common-scale px, per tile
EGO_MIN_RESPONSE = _float('EGO_MIN_RESPONSE', 0.08)
EGO_MAX_SPEED = _float('EGO_MAX_SPEED', 400.0)                # source px per frame
EGO_MAX_VELOCITY_JUMP = _float('EGO_MAX_VELOCITY_JUMP', 40.0) # px/frame vs current model
EGO_SAMPLE_WINDOW = _int('EGO_SAMPLE_WINDOW', 45)             # frames of samples kept in the fit
EGO_SAMPLE_DECAY = _float('EGO_SAMPLE_DECAY', 0.97)           # per-frame weight decay of old samples
# 'helsinki' starts from the flow fitted on the supplied scene; 'zero' from no motion.
EGO_PRIOR = _str('EGO_PRIOR', 'helsinki')
EGO_PRIOR_STRENGTH = _float('EGO_PRIOR_STRENGTH', 0.5)

# --------------------------------------------------------------------------- #
# Tracker
# --------------------------------------------------------------------------- #

# How much an observation at each level is trusted (box geometry and class vote).
LEVEL_WEIGHT = {
    0: _float('LEVEL_WEIGHT_0', 0.35),
    1: _float('LEVEL_WEIGHT_1', 0.70),
    2: _float('LEVEL_WEIGHT_2', 1.00),
}
TRACK_MATCH_MIN_IOU = _float('TRACK_MATCH_MIN_IOU', 0.05)
TRACK_MATCH_MAX_DISTANCE = _float('TRACK_MATCH_MAX_DISTANCE', 90.0)   # source px
TRACK_NEW_MIN_CONF = _float('TRACK_NEW_MIN_CONF', 0.20)
TRACK_OUTPUT_MIN_SCORE = _float('TRACK_OUTPUT_MIN_SCORE', 0.02)
# Output confidence halves every this many frames since the last observation.
TRACK_COAST_HALF_LIFE = _float('TRACK_COAST_HALF_LIFE', 12.0)
TRACK_MAX_COAST_FRAMES = _int('TRACK_MAX_COAST_FRAMES', 60)
# Looked right at it, with enough detail, and did not see it: this many times
# and the track is dropped.
TRACK_MAX_MISSES = _int('TRACK_MAX_MISSES', 2)
TRACK_TENTATIVE_HITS = _int('TRACK_TENTATIVE_HITS', 2)
TRACK_DUPLICATE_IOU = _float('TRACK_DUPLICATE_IOU', 0.45)
# Also report a track's second-most-voted class (at reduced confidence) when it
# holds at least this share of the votes. 0 disables.
TRACK_RUNNER_UP_MIN_SHARE = _float('TRACK_RUNNER_UP_MIN_SHARE', 0.2)
# A track leaves the answer once the part of its box inside the frame is thinner
# than this (source px) or a smaller share of the box than the fraction.
# Partly visible objects are scored down to a sliver, so both are permissive.
TRACK_MIN_IN_FRAME_PIXELS = _float('TRACK_MIN_IN_FRAME_PIXELS', 2.0)
TRACK_MIN_IN_FRAME_FRACTION = _float('TRACK_MIN_IN_FRAME_FRACTION', 0.0)
# Objects entering across a frame edge: their box stays pinned to that edge and
# grows while they come in, up to this multiple of the known other dimension
# (or TRACK_OPEN_MAX_EXTENT source px when both dimensions are cut, in a corner).
TRACK_OPEN_MAX_ASPECT = _float('TRACK_OPEN_MAX_ASPECT', 1.0)
TRACK_OPEN_MAX_EXTENT = _float('TRACK_OPEN_MAX_EXTENT', 200.0)

# --------------------------------------------------------------------------- #
# Camera policy
# --------------------------------------------------------------------------- #

POLICY_CELL_PIXELS = _int('POLICY_CELL_PIXELS', 120)   # coverage grid resolution
POLICY_CANDIDATE_STEP = _int('POLICY_CANDIDATE_STEP', 120)
# Detail an observation at each level provides (0..1). Viewing a cell at level L
# raises its "known detail" to this value.
POLICY_LEVEL_DETAIL = {
    0: _float('POLICY_LEVEL_DETAIL_0', 0.30),
    1: _float('POLICY_LEVEL_DETAIL_1', 0.75),
    2: _float('POLICY_LEVEL_DETAIL_2', 1.00),
}
# Known detail decays by this factor per frame, so regions become worth
# revisiting. Lower = revisit sooner.
POLICY_DETAIL_DECAY = _float('POLICY_DETAIL_DECAY', 0.93)
# Bonus for pointing at tracks that are still uncertain (class or position).
POLICY_TRACK_BONUS = _float('POLICY_TRACK_BONUS', 3.0)
# Prefer staying roughly where we are (reduces thrashing). Per 1000 px moved.
POLICY_MOVE_PENALTY = _float('POLICY_MOVE_PENALTY', 0.5)
# Fixed multiplier per level on the final score, for quick biasing.
POLICY_LEVEL_BIAS = {
    0: _float('POLICY_LEVEL_BIAS_0', 1.0),
    1: _float('POLICY_LEVEL_BIAS_1', 1.0),
    2: _float('POLICY_LEVEL_BIAS_2', 1.0),
}
# 'greedy' (value map) | 'hold_l0' (never zoom) | 'sweep_l1' (fixed patrol)
POLICY_MODE = _str('POLICY_MODE', 'sweep_l1')

# --------------------------------------------------------------------------- #
# Recording / logging
# --------------------------------------------------------------------------- #

# If set, every request's view image + metadata + our response is saved here
# (the rules allow recording the validation sequence). Written on a background
# thread so it does not cost latency.
RECORD_DIR = _str('RECORD_DIR', '')
LOG_TIMINGS = _bool('LOG_TIMINGS', True)
