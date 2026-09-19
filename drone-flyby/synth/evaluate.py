"""Compare detector checkpoints on data they were not trained on.

    python -m synth.evaluate weights/detector.pt other.pt
    python -m synth.evaluate new.pt --recording C:/path/to/merged/recording --gallery out_dir

1. Synthetic val (datasets/<set>/val): COCO mAP@0.50, per class. Its backgrounds
   are held-out LOCATIONS (never in the synthetic train split), so this measures
   transfer to new terrain with known labels.
2. --recording: a recorded validation run (read-only, evaluation only, never
   training data). No labels exist, so it reports proxies: detections per view
   by level, the class mix (one class dominating = a systematic false positive,
   like ta-ta on red roofs in the first run), and optionally a gallery of the
   highest-scoring detections per class to eyeball.

Helsinki itself is covered by debug_replay.py / run_local.py.
"""

import argparse
import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dtos import OBJECT_CLASSES  # noqa: E402
from pipeline.device import select_device  # noqa: E402


def synthetic_val(model, data_yaml, imgsz, device):
    r = model.val(data=str(data_yaml), split='val', imgsz=imgsz, batch=8, device=device, conf=0.001, iou=0.6,
                  plots=False, verbose=False)
    per_class = {model.names[int(c)]: round(float(r.box.ap50[i]), 3) for i, c in enumerate(r.box.ap_class_index)}
    return {'map50': round(float(r.box.map50), 3), 'map50_95': round(float(r.box.map), 3), 'ap50_by_class': per_class}


def recording_proxy(model, folder, imgsz, device, conf, gallery_dir=None):
    views = sorted(glob.glob(os.path.join(folder, '*.png')))
    by_level, classes, dets = Counter(), Counter(), []
    level_views = Counter()
    for png in views:
        meta = json.load(open(png[:-4] + '.json', encoding='utf-8'))
        level = meta['request']['view']['resolution_level']
        level_views[level] += 1
        r = model.predict(cv2.imread(png), imgsz=imgsz, conf=conf, iou=0.5, agnostic_nms=True, verbose=False,
                          device=device)[0]
        for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            name = model.names[int(c)]
            by_level[level] += 1
            classes[name] += 1
            dets.append((s, name, png, b, meta['request']['frame_index'], level))
    total = sum(classes.values())
    top, top_n = (classes.most_common(1)[0] if classes else (None, 0))
    out = {
        'views': len(views), 'min_conf': conf, 'detections': total,
        'per_view_by_level': {f'L{k}': round(by_level[k] / v, 2) for k, v in sorted(level_views.items())},
        'views_with_any': len({d[2] for d in dets}),
        'class_mix': dict(classes.most_common()),
        'top_class_share': round(top_n / total, 2) if total else None, 'top_class': top,
    }
    if gallery_dir:
        Path(gallery_dir).mkdir(parents=True, exist_ok=True)
        for name in classes:
            sel = sorted((d for d in dets if d[1] == name), reverse=True)[:40]
            _gallery(sel, Path(gallery_dir) / f'{name}.jpg')
    return out


def _gallery(sel, path, tile=128, cols=8, pad=24):
    rows = (len(sel) + cols - 1) // cols
    sheet = np.zeros((rows * tile, cols * tile, 3), np.uint8)
    for i, (s, name, png, b, idx, level) in enumerate(sel):
        img = cv2.imread(png)
        x1, y1, x2, y2 = (int(v) for v in b)
        cx, cy, half = (x1 + x2) // 2, (y1 + y2) // 2, max(x2 - x1, y2 - y1) // 2 + pad
        ox, oy = max(0, cx - half), max(0, cy - half)
        crop = img[oy:cy + half, ox:cx + half]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (tile, tile), interpolation=cv2.INTER_NEAREST)
        sx, sy = tile / (min(img.shape[1], cx + half) - ox), tile / (min(img.shape[0], cy + half) - oy)
        cv2.rectangle(crop, (int((x1 - ox) * sx), int((y1 - oy) * sy)), (int((x2 - ox) * sx), int((y2 - oy) * sy)), (0, 0, 255), 1)
        cv2.putText(crop, f'{s:.2f} i{idx} L{level}', (2, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
        r, c = divmod(i, cols)
        sheet[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = crop
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('weights', nargs='+')
    parser.add_argument('--synthetic', default=str(ROOT / 'datasets' / 'synth_v2' / 'data.yaml'))
    parser.add_argument('--recording', help='Folder of recorded views (one run, merged); evaluation only')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence for the recording proxy')
    parser.add_argument('--gallery', help='Write per-class crop galleries of recording detections here')
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--device', default='')
    parser.add_argument('--out', help='Write all results as JSON here')
    args = parser.parse_args()

    from ultralytics import YOLO

    device = select_device(args.device).device
    results = {}
    for w in args.weights:
        model = YOLO(w)
        res = {}
        if args.synthetic and Path(args.synthetic).is_file():
            res['synthetic_val'] = synthetic_val(model, args.synthetic, args.imgsz, device)
        if args.recording:
            gal = os.path.join(args.gallery, Path(w).stem) if args.gallery else None
            res['recording'] = recording_proxy(model, args.recording, args.imgsz, device, args.conf, gal)
        results[w] = res
        sv, rec = res.get('synthetic_val'), res.get('recording')
        print(f'== {w}')
        if sv:
            weakest = sorted(sv['ap50_by_class'].items(), key=lambda kv: kv[1])[:4]
            print(f"   synthetic val (held-out locations): mAP50 {sv['map50']}  mAP50-95 {sv['map50_95']}  weakest {weakest}")
        if rec:
            print(f"   recording @conf>={rec['min_conf']}: {rec['detections']} detections in {rec['views_with_any']}/{rec['views']} views, "
                  f"per view {rec['per_view_by_level']}, top class {rec['top_class']} {rec['top_class_share']}")
            print(f"   class mix: {rec['class_mix']}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
