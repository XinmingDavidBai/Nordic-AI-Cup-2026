"""Cut every annotated object out of the supplied scene(s) with SAM.

    python -m synth.build_cutouts                      # all scenes under src/, 6 views per object
    python -m synth.build_cutouts --per-object 10 --model sam2.1_b.pt

For each object it picks frames where the object is fully inside the frame,
spread over the sequence, crops generously around the box, upsamples the crop so
SAM sees the object large, prompts SAM with the box, and stores an RGBA PNG
cropped to the mask, at the object's original source-pixel size (the scale the
compositor needs):

    datasets/synth_assets/cutouts/<class>/<scene>_f<frame>.png
    datasets/synth_assets/cutouts/cutouts.json      class, size, mask area share, source
    datasets/synth_assets/cutouts/review.jpg        every cut-out on grey and on white, to eyeball

Only the supplied scenes are read; see synth/guard.py.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dtos import IMAGE_HEIGHT, IMAGE_WIDTH, OBJECT_CLASSES  # noqa: E402
from synth.guard import assert_training_input  # noqa: E402
from utils import DATA_DIRECTORY, frame_numbers, load_annotations, load_frame  # noqa: E402

OUT = ROOT / 'datasets' / 'synth_assets' / 'cutouts'


def pick_views(scene, per_object, margin):
    """{object_id: [(frame, bbox)]}: fully visible occurrences, spread over the sequence."""
    seen = {}
    for frame in frame_numbers(scene):
        for a in load_annotations(frame, scene):
            x1, y1, x2, y2 = a['bbox']
            if x1 < margin or y1 < margin or x2 > IMAGE_WIDTH - margin or y2 > IMAGE_HEIGHT - margin:
                continue
            seen.setdefault(a['object_id'], []).append((frame, [float(c) for c in a['bbox']]))
    picked = {}
    for name, views in seen.items():
        if len(views) <= per_object:
            picked[name] = views
        else:
            idx = np.linspace(0, len(views) - 1, per_object).round().astype(int)
            picked[name] = [views[i] for i in sorted(set(idx))]
    return picked


def segment(model, image, box, upsample_to=640, pad_frac=0.6, min_pad=12):
    """RGBA cut-out (tight to the mask) at source resolution, plus mask stats."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    pad = max(min_pad, pad_frac * max(w, h))
    cx1, cy1 = int(max(0, x1 - pad)), int(max(0, y1 - pad))
    cx2, cy2 = int(min(image.shape[1], x2 + pad)), int(min(image.shape[0], y2 + pad))
    crop = image[cy1:cy2, cx1:cx2]
    scale = upsample_to / max(crop.shape[:2])
    big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    prompt = [(x1 - cx1) * scale, (y1 - cy1) * scale, (x2 - cx1) * scale, (y2 - cy1) * scale]
    result = model(big, bboxes=[prompt], verbose=False)[0]
    if result.masks is None or len(result.masks.data) == 0:
        return None
    mask_big = result.masks.data[0].cpu().numpy().astype(np.float32)
    mask_big = cv2.resize(mask_big, (big.shape[1], big.shape[0]), interpolation=cv2.INTER_LINEAR)
    # Back to source resolution: area-average gives a soft (anti-aliased) edge.
    alpha = cv2.resize(mask_big, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_AREA)
    # Keep only the part inside the annotated box (plus a pixel): SAM sometimes
    # grabs a neighbouring shadow or patch of ground.
    keep = np.zeros_like(alpha)
    bx1, by1 = int(max(0, x1 - cx1 - 1)), int(max(0, y1 - cy1 - 1))
    bx2, by2 = int(min(crop.shape[1], x2 - cx1 + 1)), int(min(crop.shape[0], y2 - cy1 + 1))
    keep[by1:by2, bx1:bx2] = 1.0
    alpha *= keep
    box_area = max(1.0, w * h)
    share = float(alpha.sum() / box_area)
    ys, xs = np.nonzero(alpha > 0.05)
    if len(xs) == 0:
        return None
    tx1, ty1, tx2, ty2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    rgba = np.dstack([crop, (alpha * 255).clip(0, 255).astype(np.uint8)])[ty1:ty2, tx1:tx2]
    return rgba, share


def review_sheet(entries, path, tile=112):
    cols = 12
    rows = (len(entries) + cols - 1) // cols
    sheet = np.full((rows * tile * 2, cols * tile, 3), 0, np.uint8)
    for i, entry in enumerate(entries):
        rgba = cv2.imread(str(OUT / entry['file']), cv2.IMREAD_UNCHANGED)
        scale = (tile - 16) / max(rgba.shape[:2])
        small = cv2.resize(rgba, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        a = small[..., 3:4].astype(np.float32) / 255.0
        r, c = divmod(i, cols)
        for k, bg_value in enumerate((110, 255)):
            cell = np.full((tile, tile, 3), bg_value, np.uint8)
            oy, ox = (tile - small.shape[0]) // 2, (tile - small.shape[1]) // 2
            region = cell[oy:oy + small.shape[0], ox:ox + small.shape[1]].astype(np.float32)
            cell[oy:oy + small.shape[0], ox:ox + small.shape[1]] = (small[..., :3] * a + region * (1 - a)).astype(np.uint8)
            cv2.putText(cell, f"{entry['object_id'][:10]} {entry['mask_share']:.2f}", (2, 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            y0 = (2 * r + k) * tile
            sheet[y0:y0 + tile, c * tile:(c + 1) * tile] = cell
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scenes', nargs='*', help='Scenes under src/ (default: all)')
    parser.add_argument('--per-object', type=int, default=6)
    parser.add_argument('--model', default='mobile_sam.pt', help='Any ultralytics SAM checkpoint')
    parser.add_argument('--margin', type=float, default=6.0, help='Min distance (source px) from the frame edge')
    args = parser.parse_args()

    from ultralytics import SAM

    scenes = args.scenes or sorted(p.name for p in DATA_DIRECTORY.iterdir() if (p / 'images').is_dir())
    inputs = [str(assert_training_input(DATA_DIRECTORY / s)) for s in scenes]
    model = SAM(args.model)
    OUT.mkdir(parents=True, exist_ok=True)
    entries = []
    for scene in scenes:
        picked = pick_views(scene, args.per_object, args.margin)
        frames = {}
        for name in OBJECT_CLASSES:
            for frame, box in picked.get(name, []):
                if frame not in frames:
                    frames[frame] = load_frame(frame, scene)
                out = segment(model, frames[frame], box)
                if out is None:
                    print(f'  {name} f{frame}: SAM returned no mask, skipped')
                    continue
                rgba, share = out
                rel = Path(name) / f'{scene}_f{frame:06d}.png'
                (OUT / name).mkdir(exist_ok=True)
                cv2.imwrite(str(OUT / rel), rgba)
                entries.append({
                    'file': rel.as_posix(), 'object_id': name, 'scene': scene, 'frame': frame,
                    'source_bbox': [round(c, 1) for c in box],
                    'box_size_source_px': [round(box[2] - box[0], 1), round(box[3] - box[1], 1)],
                    'cutout_size_px': [int(rgba.shape[1]), int(rgba.shape[0])],
                    'mask_share': round(share, 3),
                })
                print(f'  {name:16s} f{frame:3d} box {box[2]-box[0]:5.0f}x{box[3]-box[1]:<5.0f} mask covers {share:.0%} of the box')
    # The same object across frames should cover about the same share of its box;
    # a cut-out far off its class median grabbed ground or shadow instead.
    kept, dropped = [], []
    for name in OBJECT_CLASSES:
        group = [e for e in entries if e['object_id'] == name]
        if not group:
            continue
        median = float(np.median([e['mask_share'] for e in group]))
        for e in group:
            ok = 0.6 * median <= e['mask_share'] <= 1.5 * median
            (kept if ok else dropped).append(e)
    for e in dropped:
        (OUT / e['file']).unlink(missing_ok=True)
        print(f"  dropped outlier {e['file']} (mask share {e['mask_share']})")
    entries = kept
    missing = [c for c in OBJECT_CLASSES if not any(e['object_id'] == c for e in entries)]
    manifest = {'model': args.model, 'scenes': scenes, 'inputs_checked': inputs, 'per_object': args.per_object,
                'validation_recordings_used': False, 'classes_missing': missing, 'cutouts': entries}
    (OUT / 'cutouts.json').write_text(json.dumps(manifest, indent=1))
    review_sheet(entries, OUT / 'review.jpg')
    print(f'{len(entries)} cut-outs -> {OUT}; classes missing: {missing or "none"}; review: {OUT / "review.jpg"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
