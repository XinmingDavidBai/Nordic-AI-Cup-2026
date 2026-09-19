"""Build a YOLO dataset from the supplied scenes and train a detector.

The server never sees a 4K frame: it sees 960x540 views at Level 0 (4x
downsampled), Level 1 (2x) or Level 2 (native). So the training images are
generated exactly the way the evaluator renders views (``render_view`` from
local_evaluator.py: crop, INTER_AREA resize), at all three levels, with crops
centred on objects as well as at random.

    python train_detector.py                         # build + train (yolo11n, 150 epochs)
    python train_detector.py --build-only            # just write datasets/drone_yolo
    python train_detector.py --model yolo11s.pt --epochs 300 --batch 16 --device 0
    python train_detector.py --val-frames none       # train on every frame
    python train_detector.py --extra-dataset path/to/yolo_dir   # add your own labelled data

    # quick smoke test that everything is wired (CPU, a minute or two):
    python train_detector.py --max-frames 3 --epochs 1 --imgsz 320 --l1-crops 1 --l2-crops 2

The best checkpoint is copied to weights/detector.pt, which is where the server
looks by default (DETECTOR_BACKEND=auto).

Caveat worth keeping in mind: helsinki has ONE instance per class over 25
consecutive frames, and the validation/evaluation sequences are different
scenes. Heavy augmentation matters and the val split below is leaky (adjacent
frames look alike), so treat val mAP as a smoke signal, not a promise. Add the
synthetic set (synth/README.md) with --extra-dataset. Recorded validation views
(RECORD_DIR) are evaluation-only by team rule, even labelled: build() refuses
them (synth/guard.py).
"""

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import cv2
import yaml

from dtos import OBJECT_CLASSES, SOURCE_REGION_SIZES, TRANSMITTED_VIEW_SIZE
from utils import (
    DATA_DIRECTORY,
    center_bounds_for_level,
    frame_numbers,
    load_annotations,
    load_frame,
    source_region_for_view,
)

ROOT = Path(__file__).resolve().parent
CLASS_INDEX = {name: index for index, name in enumerate(OBJECT_CLASSES)}

# Colour augmentation. 'strong' (default) because the first real validation scene
# had different terrain, season and light than helsinki and the model, which had
# only seen helsinki's colours, found almost nothing: hue/saturation/brightness
# swing hard and 10 % of images get red and blue swapped. Together with
# --gray-share (greyscale training crops) and the synthetic set's own lighting
# changes, colour stops being a cue the model can rely on.
COLOR_AUG = {
    'default': dict(hsv_h=0.015, hsv_s=0.5, hsv_v=0.4, bgr=0.0),
    'strong': dict(hsv_h=0.05, hsv_s=0.9, hsv_v=0.6, bgr=0.1),
}


def render_crop(image, level, cx, cy):
    """Identical to local_evaluator.render_view, minus the base64."""
    x1, y1, x2, y2 = source_region_for_view(level, cx, cy)
    crop = image[y1:y2, x1:x2]
    if (crop.shape[1], crop.shape[0]) != TRANSMITTED_VIEW_SIZE:
        crop = cv2.resize(crop, TRANSMITTED_VIEW_SIZE, interpolation=cv2.INTER_AREA)
    return crop, (x1, y1, x2, y2)


def yolo_labels(annotations, region, min_visible, min_view_pixels):
    x1r, y1r, x2r, y2r = region
    sx = (x2r - x1r) / TRANSMITTED_VIEW_SIZE[0]
    sy = (y2r - y1r) / TRANSMITTED_VIEW_SIZE[1]
    lines = []
    for annotation in annotations:
        bx1, by1, bx2, by2 = (float(c) for c in annotation['bbox'])
        area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        cx1, cy1 = max(bx1, x1r), max(by1, y1r)
        cx2, cy2 = min(bx2, x2r), min(by2, y2r)
        if cx2 <= cx1 or cy2 <= cy1 or area <= 0:
            continue
        if (cx2 - cx1) * (cy2 - cy1) / area < min_visible:
            continue
        w = (cx2 - cx1) / sx
        h = (cy2 - cy1) / sy
        if w < min_view_pixels or h < min_view_pixels:
            continue
        xc = ((cx1 + cx2) / 2 - x1r) / sx / TRANSMITTED_VIEW_SIZE[0]
        yc = ((cy1 + cy2) / 2 - y1r) / sy / TRANSMITTED_VIEW_SIZE[1]
        lines.append(
            f'{CLASS_INDEX[annotation["object_id"]]} {xc:.6f} {yc:.6f} '
            f'{w / TRANSMITTED_VIEW_SIZE[0]:.6f} {h / TRANSMITTED_VIEW_SIZE[1]:.6f}'
        )
    return lines


def crop_centres(level, annotations, count, object_share, rng):
    """Random legal centres, a share of them placed so an object is in view."""
    min_x, max_x, min_y, max_y = center_bounds_for_level(level)
    width, height = SOURCE_REGION_SIZES[level]
    centres = []
    for index in range(count):
        if annotations and index < round(count * object_share):
            bx1, by1, bx2, by2 = rng.choice(annotations)['bbox']
            ox, oy = (bx1 + bx2) / 2, (by1 + by2) / 2
            # Anywhere in the view, not always dead centre.
            cx = ox + rng.uniform(-0.4, 0.4) * width
            cy = oy + rng.uniform(-0.4, 0.4) * height
        else:
            cx, cy = rng.uniform(min_x, max_x), rng.uniform(min_y, max_y)
        centres.append((int(min(max(cx, min_x), max_x)), int(min(max(cy, min_y), max_y))))
    return centres


def build(args) -> Path:
    from synth.guard import assert_training_input

    # Recorded validation views are for evaluation only (team rule). Checked
    # before anything is deleted or built, so a bad path fails immediately.
    scenes = args.scenes or sorted(p.name for p in DATA_DIRECTORY.iterdir() if (p / 'images').is_dir())
    for path in [DATA_DIRECTORY / scene for scene in scenes] + list(args.extra_dataset or []):
        assert_training_input(path)

    rng = random.Random(args.seed)
    out = Path(args.dataset_dir)
    if out.exists():
        shutil.rmtree(out)
    for split in ('train', 'val'):
        (out / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out / 'labels' / split).mkdir(parents=True, exist_ok=True)

    counts = {'train': 0, 'val': 0}
    instances = {'train': 0, 'val': 0}
    for scene in scenes:
        frames = frame_numbers(scene)
        if args.max_frames:
            frames = frames[: args.max_frames]
        if args.val_frames == 'none':
            val = set()
        elif args.val_frames == 'auto':
            val = set(frames[::6][1:]) if len(frames) > 6 else set()
        else:
            val = {int(v) for v in args.val_frames.split(',') if v.strip()}

        for frame in frames:
            split = 'val' if frame in val else 'train'
            image = load_frame(frame, scene)
            annotations = load_annotations(frame, scene)
            jobs = [(0, 1920, 1080)]
            jobs += [(1, cx, cy) for cx, cy in crop_centres(1, annotations, args.l1_crops, args.object_share, rng)]
            jobs += [(2, cx, cy) for cx, cy in crop_centres(2, annotations, args.l2_crops, args.object_share, rng)]
            for index, (level, cx, cy) in enumerate(jobs):
                crop, region = render_crop(image, level, cx, cy)
                labels = yolo_labels(annotations, region, args.min_visible, args.min_view_pixels)
                if not labels and rng.random() > args.keep_empty:
                    continue
                stem = f'{scene}_f{frame:06d}_L{level}_{index:02d}'
                # Colour does not transfer between scenes; make the model do without it.
                # (gray_share 0 draws no random number, so it rebuilds exactly the old dataset.)
                if split == 'train' and args.gray_share and rng.random() < args.gray_share:
                    crop = cv2.cvtColor(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
                cv2.imwrite(str(out / 'images' / split / f'{stem}.png'), crop, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                (out / 'labels' / split / f'{stem}.txt').write_text('\n'.join(labels) + ('\n' if labels else ''))
                counts[split] += 1
                instances[split] += len(labels)
        print(f'{scene}: {len(frames)} frames, val frames {sorted(val)}')

    train_dirs = [str((out / 'images' / 'train').resolve())]
    for extra in args.extra_dataset or []:
        extra_images = Path(extra) / 'images'
        train_dirs.append(str((extra_images if extra_images.is_dir() else Path(extra)).resolve()))
    val_dir = (out / 'images' / 'val').resolve()
    if counts['val'] == 0:
        val_dir = (out / 'images' / 'train').resolve()   # ultralytics needs something
    data = {
        'train': train_dirs if len(train_dirs) > 1 else train_dirs[0],
        'val': str(val_dir),
        'names': {index: name for index, name in enumerate(OBJECT_CLASSES)},
    }
    data_yaml = out / 'data.yaml'
    with open(data_yaml, 'w') as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    print(f'dataset: {counts["train"]} train images ({instances["train"]} boxes), '
          f'{counts["val"]} val images ({instances["val"]} boxes) -> {data_yaml}')
    return data_yaml


def train(args, data_yaml: Path) -> None:
    from ultralytics import YOLO

    from pipeline.device import select_device

    # Same rules as serving: GPU whenever torch can use one; a forced GPU it
    # cannot see is an error here rather than a slow CPU run.
    choice = select_device(args.device)
    device = choice.device
    print(f'training {args.model}: {choice.summary()} (torch {choice.torch_version}, {choice.torch_build})')
    if choice.kind == 'cpu' and choice.how != 'forced':
        banner = '!' * 78
        print(f'{banner}\n!!! TRAINING ON THE CPU: {choice.reason}\n'
              f'!!! That is hours instead of minutes. See PIPELINE.md section 1a for GPU torch,\n'
              f'!!! or pass --device cpu to do this on purpose.\n{banner}', flush=True)
    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=device,
        workers=args.workers,
        project=str(Path(args.project).resolve()),
        name=args.name,
        exist_ok=True,
        patience=args.patience,
        seed=args.seed,
        # Top-down imagery: any flip is a plausible view. Small scale jitter
        # stands in for altitude/level variation beyond the three levels.
        fliplr=0.5,
        flipud=0.5,
        degrees=0.0,      # rotation re-boxes objects loosely; flips are exact
        scale=0.3,
        translate=0.1,
        mosaic=1.0,
        close_mosaic=max(1, min(10, args.epochs // 5)),
        **COLOR_AUG[args.color_aug],
        plots=True,
    )
    best, last = Path(model.trainer.best), Path(model.trainer.last)
    print(f'best checkpoint: {best}')
    if args.no_install or not best.is_file():
        return
    fitness = float(model.trainer.best_fitness or 0.0)
    print(f'best fitness (ultralytics accuracy score for this run): {fitness:.4f}')

    # Keep 4 files total: the best.pt and last.pt of this run and of the
    # previous one (a rolling 2-run window), tracked in a manifest alongside
    # them since each run is a fresh process. Whichever run's best.pt has the
    # higher fitness -- not just whichever ran most recently -- becomes
    # weights/detector.pt, what the server loads; last.pt is kept for
    # reference/resuming only and never competes for that slot.
    archive_dir = ROOT / 'models_weights'
    archive_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = archive_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else []

    stamp = time.strftime('%Y%m%d_%H%M%S')
    best_file = f'{args.name}_{stamp}_best.pt'
    last_file = f'{args.name}_{stamp}_last.pt'
    shutil.copy2(best, archive_dir / best_file)
    if last.is_file():
        shutil.copy2(last, archive_dir / last_file)
    else:
        last_file = None
    manifest.insert(0, {
        'name': args.name, 'timestamp': stamp, 'fitness': fitness,
        'best_file': best_file, 'last_file': last_file,
    })
    print(f'archived  -> {archive_dir / best_file}' + (f' and {archive_dir / last_file}' if last_file else ''))

    while len(manifest) > 2:
        stale = manifest.pop()
        for key in ('best_file', 'last_file'):
            stale_path = archive_dir / stale[key] if stale[key] else None
            if stale_path and stale_path.is_file():
                stale_path.unlink()
                print(f'pruned    -> {stale_path.name} (older than the last 2 runs)')

    manifest_path.write_text(json.dumps(manifest, indent=2))

    primary = max(manifest, key=lambda entry: entry['fitness'])
    target = ROOT / 'weights' / 'detector.pt'
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(archive_dir / primary['best_file'], target)
    print(f'installed -> {target}  (from {primary["best_file"]}, fitness {primary["fitness"]:.4f}; '
          f'the server picks it up with DETECTOR_BACKEND=auto)')
    for entry in manifest:
        role = 'primary' if entry is primary else 'secondary'
        print(f'  kept ({role}): best={entry["best_file"]}  last={entry["last_file"]}  '
              f'fitness={entry["fitness"]:.4f}  run={entry["name"]}')

    results_png = Path(model.trainer.save_dir) / 'results.png'
    if results_png.is_file():
        print(f'training curves -> {results_png}')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scenes', nargs='*', help='Scenes under src/ (default: all)')
    parser.add_argument('--dataset-dir', default=str(ROOT / 'datasets' / 'drone_yolo'))
    parser.add_argument('--val-frames', default='auto', help="'auto' (every 6th frame), 'none', or e.g. '4,12,20'")
    parser.add_argument('--max-frames', type=int, default=0, help='Use only the first N frames per scene')
    parser.add_argument('--l1-crops', type=int, default=6, help='Level-1 views per frame')
    parser.add_argument('--l2-crops', type=int, default=12, help='Level-2 views per frame')
    parser.add_argument('--object-share', type=float, default=0.75, help='Share of crops centred near an object')
    parser.add_argument('--min-visible', type=float, default=0.4, help='Min share of a box inside the crop to label it')
    parser.add_argument('--min-view-pixels', type=float, default=2.0)
    parser.add_argument('--keep-empty', type=float, default=0.3, help='Probability of keeping a crop with no objects')
    parser.add_argument('--extra-dataset', action='append',
                        help='Extra YOLO-format dir (images/ + labels/) for training, e.g. datasets/synth_v2/train. '
                             'Recorded validation views are refused (synth/guard.py)')
    parser.add_argument('--color-aug', choices=sorted(COLOR_AUG), default='strong',
                        help="Colour augmentation profile (default 'strong'; 'default' = the old settings)")
    parser.add_argument('--gray-share', type=float, default=0.2,
                        help='Share of generated TRAIN crops written greyscale (val untouched)')
    parser.add_argument('--build-only', action='store_true')
    parser.add_argument('--model', default='yolo11n.pt', help='Any ultralytics detection checkpoint or yaml')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--device', default='',
                        help="'' auto (GPU whenever torch can use one), 'cpu', '0' / 'cuda:N' (CUDA or ROCm GPU), 'mps'")
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--project', default=str(ROOT / 'runs'))
    parser.add_argument('--name', default='drone_detector')
    parser.add_argument('--no-install', action='store_true', help="Don't copy best.pt to weights/detector.pt")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    data_yaml = build(args)
    if not args.build_only:
        train(args, data_yaml)
    return 0


if __name__ == '__main__':
    sys.exit(main())
