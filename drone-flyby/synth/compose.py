"""Composite object cut-outs onto aerial backgrounds, rendered like the evaluator.

    python -m synth.compose                               # datasets/synth_v2: 6000 train + 600 val views
    python -m synth.compose --resume                      # finish an interrupted run (same arguments)
    python -m synth.compose --train 24 --val 8 --name synth_smoke   # quick look

Each image is made the way the evaluator makes a view: build a SOURCE-resolution
canvas for the level (3840x2160 at L0, 1920x1080 at L1, 960x540 at L2), paste
objects at their real source-pixel size, then INTER_AREA it down to 960x540.

Per image:
- background: a random crop of an aerial mosaic, scaled so one source pixel is
  ~0.12-0.26 m (the drone's scale), random 90-degree turn / flip, mild colour shift;
- objects: 0-8 cut-outs, classes balanced, any rotation (the box is re-fitted
  tightly to the rotated mask, so rotation stays exact), +-15 % size, flips,
  brightness / contrast / saturation / hue jitter, a soft drop shadow, a little
  blur and noise; some placed across the view edge (labelled if >= 40 % visible,
  like train_detector.py);
- lighting: whole-image white balance, gamma, hue and saturation shifts, and
  25 % of images greyscale (--gray-share), so colour and lighting are not cues
  the model can lean on (they did not transfer to the validation scene);
- ~12 % of images get no objects at all: towns, cars, roofs and rails with no
  labels are exactly the hard negatives the validation run showed we need.

Output (YOLO layout, class ids in dtos.OBJECT_CLASSES order):

    datasets/<name>/train/images, train/labels     -> train_detector.py --extra-dataset datasets/<name>/train
    datasets/<name>/val/images, val/labels         held-out BACKGROUND LOCATIONS, never trained on
    datasets/<name>/data.yaml                      train/val of the synthetic set alone (for evaluation)
    datasets/<name>/manifest.json                  seed, args, per-class/level counts, input hashes
    datasets/<name>/preview.jpg                    a grid of samples with their boxes

Inputs are the cut-out bank (synth/build_cutouts.py) and backgrounds
(synth/fetch_backgrounds.py). Both roots are checked by synth/guard.py: recorded
validation views are never read.
"""

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dtos import OBJECT_CLASSES, SOURCE_REGION_SIZES, TRANSMITTED_VIEW_SIZE  # noqa: E402
from synth.fetch_backgrounds import EXCLUDE_BBOX  # noqa: E402
from synth.guard import assert_training_input  # noqa: E402

ASSETS = ROOT / 'datasets' / 'synth_assets'
CLASS_INDEX = {name: i for i, name in enumerate(OBJECT_CLASSES)}
VIEW_W, VIEW_H = TRANSMITTED_VIEW_SIZE


# --------------------------------------------------------------------------- #
# Assets
# --------------------------------------------------------------------------- #

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load_cutouts(root: Path):
    manifest = json.loads((root / 'cutouts.json').read_text(encoding='utf-8'))
    bank = {}
    for e in manifest['cutouts']:
        rgba = cv2.imread(str(root / e['file']), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.shape[2] != 4:
            continue
        bank.setdefault(e['object_id'], []).append((e['file'], rgba))
    missing = [c for c in OBJECT_CLASSES if c not in bank]
    if missing:
        raise SystemExit(f'cut-out bank has no {missing}; run python -m synth.build_cutouts')
    return bank, manifest


def load_backgrounds(root: Path):
    index = json.loads((root / 'backgrounds.json').read_text(encoding='utf-8'))
    by_split = {'train': [], 'val': []}
    lon0, lat0, lon1, lat1 = (float(v) for v in EXCLUDE_BBOX.split(','))
    for e in index['images']:
        if lon0 <= float(e['lon']) <= lon1 and lat0 <= float(e['lat']) <= lat1:
            raise SystemExit(f"background {e['file']} ({e['title']!r}) lies in the excluded validation area "
                             f'{EXCLUDE_BBOX}; remove it (see synth/fetch_backgrounds.py)')
        if (root / e['file']).is_file():
            by_split[e['split']].append(e)
    if not by_split['train'] or not by_split['val']:
        raise SystemExit(f'need train and val backgrounds, have {[len(v) for v in by_split.values()]}; '
                         'run python -m synth.fetch_backgrounds')
    return by_split, index


class BackgroundCache:
    def __init__(self, root, size=24):
        self.root, self.size, self.cache = root, size, {}

    def get(self, entry):
        name = entry['file']
        if name not in self.cache:
            if len(self.cache) >= self.size:
                self.cache.pop(next(iter(self.cache)))
            self.cache[name] = cv2.imread(str(self.root / name), cv2.IMREAD_COLOR)
        return self.cache[name]


# --------------------------------------------------------------------------- #
# Image pieces
# --------------------------------------------------------------------------- #

def background_canvas(rng, cache, entry, level):
    """Source-resolution canvas for one view at ``level``; returns (canvas, m_per_source_px)."""
    mosaic = cache.get(entry)
    mpp = float(entry['m_per_px'])
    w, h = SOURCE_REGION_SIZES[level]
    turns = rng.randrange(4)
    if turns % 2:
        mosaic = np.ascontiguousarray(np.rot90(mosaic, turns))
    elif turns:
        mosaic = np.ascontiguousarray(np.rot90(mosaic, turns))
    mh, mw = mosaic.shape[:2]
    target = rng.uniform(0.12, 0.26)                   # metres per SOURCE pixel we want
    target = min(target, mw * mpp / w, mh * mpp / h)   # the mosaic must cover the view
    crop_w, crop_h = int(w * target / mpp), int(h * target / mpp)
    x0, y0 = rng.randint(0, mw - crop_w), rng.randint(0, mh - crop_h)
    crop = mosaic[y0:y0 + crop_h, x0:x0 + crop_w]
    interp = cv2.INTER_AREA if crop_w > w else cv2.INTER_LINEAR
    canvas = cv2.resize(crop, (w, h), interpolation=interp)
    if rng.random() < 0.5:
        canvas = canvas[:, ::-1]
    canvas = jitter_colour(rng, canvas.astype(np.float32), strength=1.0)
    return canvas, target


def jitter_colour(rng, img, strength=1.0):
    """Brightness / contrast / saturation / hue jitter on a float32 BGR image."""
    b = 1.0 + rng.uniform(-0.22, 0.18) * strength
    c = 1.0 + rng.uniform(-0.2, 0.2) * strength
    mean = img.mean(axis=(0, 1), keepdims=True)
    img = (img - mean) * c + mean * b
    hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-4, 4) * strength) % 180
    hsv[..., 1] *= 1.0 + rng.uniform(-0.25, 0.2) * strength
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)


def transform_object(rng, rgba):
    """Scale / flip / rotate a cut-out; returns (bgr float32, alpha float32 0..1)."""
    scale = rng.uniform(0.85, 1.15)
    rgba = cv2.resize(rgba, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    if rng.random() < 0.5:
        rgba = rgba[:, ::-1]
    angle = rng.uniform(0, 360)
    h, w = rgba.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(math.ceil(h * sin + w * cos)) + 2, int(math.ceil(h * cos + w * sin)) + 2
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    # Rotate premultiplied colour so the transparent surround does not bleed in.
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    pre = np.dstack([rgba[..., :3].astype(np.float32) * a, a[..., 0]])
    out = cv2.warpAffine(pre, m, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0)
    alpha = np.clip(out[..., 3], 0, 1)
    bgr = out[..., :3] / np.maximum(alpha[..., None], 1e-4)
    bgr = jitter_colour(rng, bgr, strength=1.0)
    ys, xs = np.nonzero(alpha > 0.35)
    if len(xs) == 0:
        return None
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    return bgr[y1:y2, x1:x2], alpha[y1:y2, x1:x2]


def paste(canvas, bgr, alpha, x, y, shadow):
    """Alpha-composite at (x, y) (may hang over the edge); returns the visible box or None."""
    H, W = canvas.shape[:2]
    h, w = alpha.shape
    if shadow is not None:
        dx, dy, strength, blur = shadow
        s = cv2.GaussianBlur(alpha, (0, 0), blur) * strength
        _blend(canvas, np.zeros_like(bgr), s, x + dx, y + dy)
    _blend(canvas, bgr, alpha, x, y)
    ys, xs = np.nonzero(alpha > 0.35)
    bx1, by1, bx2, by2 = x + xs.min(), y + ys.min(), x + xs.max() + 1, y + ys.max() + 1
    full = (bx2 - bx1) * (by2 - by1)
    vx1, vy1, vx2, vy2 = max(bx1, 0), max(by1, 0), min(bx2, W), min(by2, H)
    if vx2 <= vx1 or vy2 <= vy1:
        return None, 0.0
    return (vx1, vy1, vx2, vy2), (vx2 - vx1) * (vy2 - vy1) / full


def _blend(canvas, bgr, alpha, x, y):
    H, W = canvas.shape[:2]
    h, w = alpha.shape
    x1, y1, x2, y2 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x2 <= x1 or y2 <= y1:
        return
    a = alpha[y1 - y:y2 - y, x1 - x:x2 - x, None]
    region = canvas[y1:y2, x1:x2]
    canvas[y1:y2, x1:x2] = region * (1 - a) + bgr[y1 - y:y2 - y, x1 - x:x2 - x] * a


def overlaps(box, boxes, limit=0.05):
    for b in boxes:
        ix = max(0, min(box[2], b[2]) - max(box[0], b[0]))
        iy = max(0, min(box[3], b[3]) - max(box[1], b[1]))
        inter = ix * iy
        if inter / ((box[2] - box[0]) * (box[3] - box[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter) > limit:
            return True
    return False


# --------------------------------------------------------------------------- #
# One image
# --------------------------------------------------------------------------- #

def make_image(rng, bank, cache, backgrounds, args, class_cycle):
    level = rng.choices((0, 1, 2), weights=args.level_weights)[0]
    entry = rng.choice(backgrounds)
    canvas, _ = background_canvas(rng, cache, entry, level)
    H, W = canvas.shape[:2]
    view_scale = W / VIEW_W                            # source px per view px
    n = 0 if rng.random() < args.empty_share else rng.choice(args.objects_choices)
    sun = rng.uniform(0, 2 * math.pi)
    shadow_len = rng.uniform(1.5, 6.0)
    labels, placed, used = [], [], []
    for _ in range(n):
        name = next(class_cycle)
        file, rgba = rng.choice(bank[name])
        obj = transform_object(rng, rgba)
        if obj is None:
            continue
        bgr, alpha = obj
        h, w = alpha.shape
        across_edge = rng.random() < args.edge_share
        for _attempt in range(20):
            if across_edge:
                x = rng.randint(-w // 2, W - w // 2)
                y = rng.choice([rng.randint(-h // 2, 0), rng.randint(H - h, H - h // 2)]) if rng.random() < 0.5 \
                    else rng.randint(-h // 2, H - h // 2)
                if rng.random() < 0.5:
                    x = rng.choice([rng.randint(-w // 2, 0), rng.randint(W - w, W - w // 2)])
            else:
                x, y = rng.randint(0, max(0, W - w)), rng.randint(0, max(0, H - h))
            box = (x, y, x + w, y + h)
            if not overlaps(box, placed):
                break
        else:
            continue
        shadow = None
        if rng.random() < 0.8:
            shadow = (int(round(math.cos(sun) * shadow_len)), int(round(math.sin(sun) * shadow_len)),
                      rng.uniform(0.25, 0.5), rng.uniform(1.0, 2.5))
        visible, share = paste(canvas, bgr, alpha, x, y, shadow)
        placed.append(box)
        if visible is None or share < args.min_visible:
            continue
        vw, vh = (visible[2] - visible[0]) / view_scale, (visible[3] - visible[1]) / view_scale
        if vw < args.min_view_pixels or vh < args.min_view_pixels:
            continue
        cx = (visible[0] + visible[2]) / 2 / W
        cy = (visible[1] + visible[3]) / 2 / H
        labels.append(f'{CLASS_INDEX[name]} {cx:.6f} {cy:.6f} {vw / VIEW_W:.6f} {vh / VIEW_H:.6f}')
        used.append((name, file))
    if rng.random() < 0.35:
        canvas = cv2.GaussianBlur(canvas, (0, 0), rng.uniform(0.3, 1.0))
    view = cv2.resize(np.clip(canvas, 0, 255).astype(np.uint8), (VIEW_W, VIEW_H), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.3:
        noise = np.random.default_rng(rng.randrange(2 ** 31)).normal(0, rng.uniform(1, 3.5), view.shape)
        view = np.clip(view.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    view = scene_lighting(rng, view, args)
    return view, labels, level, entry['file'], used


def scene_lighting(rng, view, args):
    """Whole-image lighting/colour change, so colour is not a cue that transfers.

    White balance (per-channel gain), gamma and a global hue turn, then with
    probability --gray-share the image goes greyscale (kept 3-channel: the
    detector always gets BGR).
    """
    img = view.astype(np.float32) / 255.0
    s = args.color_strength
    gains = np.array([rng.uniform(1 - 0.18 * s, 1 + 0.18 * s) for _ in range(3)], np.float32)
    img = np.clip(img * gains, 0, 1) ** rng.uniform(1 / (1 + 0.45 * s), 1 + 0.45 * s)
    out = (img * 255).astype(np.uint8)
    if s > 0 and rng.random() < 0.5:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
        hsv[..., 0] = (hsv[..., 0].astype(np.int16) + int(rng.uniform(-12, 12) * s)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1].astype(np.float32) * rng.uniform(1 - 0.5 * s, 1 + 0.3 * s), 0, 255)
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if rng.random() < args.gray_share:
        out = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return out


def class_cycler(rng):
    while True:
        order = list(OBJECT_CLASSES)
        rng.shuffle(order)
        yield from order


_W = {}   # per worker process: bank, backgrounds, cache, args, out


def _init_worker(bank, backgrounds, bg_root, args, out):
    cv2.setNumThreads(1)
    _W.update(bank=bank, backgrounds=backgrounds, cache=BackgroundCache(bg_root), args=args, out=out)


def _render_one(task):
    """One image. Its own RNG seeded from (seed, split, index): the same dataset
    comes out whatever the number of workers."""
    split, i = task
    args, out = _W['args'], _W['out']
    if args.resume:
        # The label file is written last, so its presence means the image is complete.
        done = list((out / split / 'labels').glob(f'{args.name}_{split}_{i:06d}_L*.txt'))
        if done:
            text = done[0].read_text()
            return split, [line for line in text.split('\n') if line.strip()], int(done[0].stem[-1]), None
    rng = random.Random(f'{args.seed}:{split}:{i}')
    view, labels, level, bg, used = make_image(rng, _W['bank'], _W['cache'], _W['backgrounds'][split], args,
                                               class_cycler(rng))
    stem = f'{args.name}_{split}_{i:06d}_L{level}'
    if args.jpg:
        cv2.imwrite(str(out / split / 'images' / f'{stem}.jpg'), view, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        cv2.imwrite(str(out / split / 'images' / f'{stem}.png'), view, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    (out / split / 'labels' / f'{stem}.txt').write_text('\n'.join(labels) + ('\n' if labels else ''))
    return split, labels, level, bg


def write_all(counts, bank, backgrounds, bg_root, args, out, stats):
    from multiprocessing import Pool

    for split in counts:
        (out / split / 'images').mkdir(parents=True, exist_ok=True)
        (out / split / 'labels').mkdir(parents=True, exist_ok=True)
    tasks = [(split, i) for split, n in counts.items() for i in range(n)]
    started, done = time.time(), 0
    with Pool(args.workers, initializer=_init_worker, initargs=(bank, backgrounds, bg_root, args, out)) as pool:
        for split, labels, level, bg in pool.imap_unordered(_render_one, tasks, chunksize=8):
            s = stats[split]
            s['images'] += 1
            s['boxes'] += len(labels)
            s['empty'] += not labels
            s['levels'][f'L{level}'] += 1
            if bg is not None:          # None: kept from an earlier, interrupted run (--resume)
                s['backgrounds'][bg] += 1
            for line in labels:
                s['classes'][OBJECT_CLASSES[int(line.split()[0])]] += 1
            done += 1
            if done % 500 == 0:
                print(f'  {done}/{len(tasks)} ({done / (time.time() - started):.1f} img/s)', flush=True)


def preview(out, path, n=24, tile=(320, 180)):
    files = sorted((out / 'train' / 'images').iterdir())[:n]
    cols = 6
    rows = math.ceil(len(files) / cols)
    sheet = np.zeros((rows * tile[1], cols * tile[0], 3), np.uint8)
    for i, f in enumerate(files):
        img = cv2.imread(str(f))
        for line in (out / 'train' / 'labels' / (f.stem + '.txt')).read_text().split('\n'):
            if not line.strip():
                continue
            c, cx, cy, w, h = line.split()
            cx, cy, w, h = float(cx) * VIEW_W, float(cy) * VIEW_H, float(w) * VIEW_W, float(h) * VIEW_H
            cv2.rectangle(img, (int(cx - w / 2), int(cy - h / 2)), (int(cx + w / 2), int(cy + h / 2)), (0, 0, 255), 2)
            cv2.putText(img, OBJECT_CLASSES[int(c)], (int(cx - w / 2), max(10, int(cy - h / 2) - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        r, c = divmod(i, cols)
        sheet[r * tile[1]:(r + 1) * tile[1], c * tile[0]:(c + 1) * tile[0]] = cv2.resize(img, tile, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--name', default='synth_v2')
    parser.add_argument('--train', type=int, default=6000)
    parser.add_argument('--val', type=int, default=600)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cutouts', default=str(ASSETS / 'cutouts'))
    parser.add_argument('--backgrounds', default=str(ASSETS / 'backgrounds'))
    parser.add_argument('--level-weights', type=float, nargs=3, default=[0.3, 0.5, 0.2], metavar=('L0', 'L1', 'L2'))
    parser.add_argument('--empty-share', type=float, default=0.12)
    parser.add_argument('--objects-choices', type=int, nargs='+', default=[1, 2, 2, 3, 3, 4, 5, 6, 8])
    parser.add_argument('--edge-share', type=float, default=0.12)
    parser.add_argument('--min-visible', type=float, default=0.4)
    parser.add_argument('--min-view-pixels', type=float, default=2.0)
    parser.add_argument('--gray-share', type=float, default=0.25,
                        help='Share of images turned greyscale, so the model cannot rely on colour')
    parser.add_argument('--color-strength', type=float, default=1.0,
                        help='Scale of the whole-image white-balance / gamma / hue / saturation change (0 = off)')
    parser.add_argument('--jpg', action='store_true', help='JPEG q95 instead of PNG (much smaller, slightly lossy)')
    parser.add_argument('--workers', type=int, default=max(1, min(12, (__import__('os').cpu_count() or 2) - 2)))
    parser.add_argument('--resume', action='store_true',
                        help='Keep images an interrupted run already wrote (same arguments!) and render only the rest')
    args = parser.parse_args()

    cut_root = assert_training_input(args.cutouts)
    bg_root = assert_training_input(args.backgrounds)
    bank, cut_manifest = load_cutouts(cut_root)
    backgrounds, bg_index = load_backgrounds(bg_root)
    if cut_manifest.get('validation_recordings_used') or bg_index.get('validation_recordings_used'):
        raise SystemExit('an input manifest says it used validation recordings; refusing')

    out = ROOT / 'datasets' / args.name
    if out.exists() and not args.resume:
        import shutil
        shutil.rmtree(out)
    stats = {s: {'images': 0, 'boxes': 0, 'empty': 0, 'levels': Counter(), 'classes': Counter(), 'backgrounds': Counter()}
             for s in ('train', 'val')}
    # Train images only ever use train-split background locations, val images only
    # val-split ones, so the synthetic val set measures places never trained on.
    write_all({'train': args.train, 'val': args.val}, bank, backgrounds, bg_root, args, out, stats)

    data = {'train': str((out / 'train' / 'images').resolve()), 'val': str((out / 'val' / 'images').resolve()),
            'names': {i: n for i, n in enumerate(OBJECT_CLASSES)}}
    (out / 'data.yaml').write_text(yaml.safe_dump(data, sort_keys=False))
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = None
    manifest = {
        'name': args.name, 'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'git_commit': commit, 'args': vars(args),
        'validation_recordings_used': False,
        'inputs_checked': {'cutouts': str(cut_root), 'backgrounds': str(bg_root)},
        'cutout_files': {name: sorted({f for f, _ in items}) for name, items in bank.items()},
        'cutouts_json_sha': _sha(cut_root / 'cutouts.json'),
        'backgrounds': {s: [e['file'] for e in backgrounds[s]] for s in ('train', 'val')},
        'background_attribution': [f"{e['title']} ({e['provider']}), OpenAerialMap, CC-BY 4.0"
                                   for s in ('train', 'val') for e in backgrounds[s]],
        'stats': {s: {k: (dict(v) if isinstance(v, Counter) else v) for k, v in st.items() if k != 'backgrounds'}
                  for s, st in stats.items()},
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding='utf-8')
    preview(out, out / 'preview.jpg')
    for s in ('train', 'val'):
        st = stats[s]
        print(f"{s}: {st['images']} images, {st['boxes']} boxes, {st['empty']} empty, levels {dict(st['levels'])}, "
              f"{len(st['backgrounds'])} background locations")
    print(f'-> {out}  (train with: python train_detector.py --extra-dataset {out / "train"})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
