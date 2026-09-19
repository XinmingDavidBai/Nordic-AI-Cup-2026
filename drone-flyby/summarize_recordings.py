"""Boil a recorded validation run down to a small summary you can send around.

A validation run recorded by api.py (``recordings/<sequence_id>/``) is hundreds
of MB of PNGs. This reads it where it lies and writes a few hundred KB:

    python summarize_recordings.py                               # newest run in recordings/
    python summarize_recordings.py recordings/<sequence_id>
    python summarize_recordings.py --log server.log              # also parse the server log
    python summarize_recordings.py --detect                      # also run the detector on every view
    python summarize_recordings.py --out run_summary --total-frames 249

Output folder (default ``run_summary/``), small enough to paste or zip:

    summary.txt         the human-readable report
    summary.json        the same numbers, machine-readable
    contact_sheet.jpg   a grid of downscaled views, to eyeball the scene and object sizes

What it answers:

- Coverage and timing: which frame_index values arrived, how many were skipped
  (a skipped frame scores zero detections, so this caps the score), and the
  time between requests (from the arrival timestamps in newer recordings).
- Protocol health: repeated request_ids, out-of-order frame_index, several
  process ids (several servers answering one run), refused camera moves.
- What we answered: annotations per frame, per class, confidence spread,
  empty-answer streaks.
- Object sizes: answered box sizes per class in source pixels and in the view
  pixels they would have at levels 0/1/2 (compare with the detector's comfort
  zone). With --detect: the raw detector output on every recorded view, in view
  pixels, which does not depend on the tracker.
- With --log: the pipeline's own per-frame timings and every warning/error kind.

No ground truth exists for validation, so this cannot compute a score. It shows
where one is being lost.
"""

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_WIDTH, SOURCE_HEIGHT = 3840, 2160
FRAME_INTERVAL_MS = 333.0
LEVEL_SCALE = {0: 4.0, 1: 2.0, 2: 1.0}   # source px per transmitted view px


def _stats(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None

    def pct(p):
        return values[min(len(values) - 1, int(round(p * (len(values) - 1))))]

    return {
        'n': len(values),
        'min': round(values[0], 2),
        'p10': round(pct(0.10), 2),
        'median': round(statistics.median(values), 2),
        'mean': round(statistics.fmean(values), 2),
        'p90': round(pct(0.90), 2),
        'max': round(values[-1], 2),
    }


def _fmt(s, unit=''):
    if not s:
        return 'n/a'
    return (f"median {s['median']}{unit}, mean {s['mean']}{unit}, "
            f"p10-p90 {s['p10']}-{s['p90']}{unit}, min {s['min']}{unit}, max {s['max']}{unit} (n={s['n']})")


def find_run(path: Path) -> Path:
    """A sequence folder, or the most recently modified one under a recordings root."""
    if any(path.glob('*.json')):
        return path
    runs = [p for p in path.iterdir() if p.is_dir() and any(p.glob('*.json'))]
    if not runs:
        raise SystemExit(f'No recorded requests under {path}')
    return max(runs, key=lambda p: max(f.stat().st_mtime for f in p.glob('*.json')))


def load_records(run: Path):
    records = []
    for path in run.glob('*.json'):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except Exception as error:
            print(f'skipping unreadable {path.name}: {error}', file=sys.stderr)
            continue
        data['_file'] = path.name
        records.append(data)
    # Newer recordings carry the arrival order; older ones only frame_index.
    records.sort(key=lambda r: (r.get('meta', {}).get('arrival', 0), r['request']['frame_index']))
    return records


def summarize(records, total_frames):
    requests = [r['request'] for r in records]
    responses = [r.get('response') for r in records]
    metas = [r.get('meta', {}) for r in records]
    out = {'requests_recorded': len(records)}

    # ---- protocol health -------------------------------------------------- #
    ids = Counter(q['request_id'] for q in requests)
    out['sequence_ids'] = sorted({q['sequence_id'] for q in requests})
    out['repeated_request_ids'] = sum(c - 1 for c in ids.values() if c > 1)
    out_of_order, highest = [], -1
    for q in requests:
        if q['frame_index'] <= highest:
            out_of_order.append(q['frame_index'])
        highest = max(highest, q['frame_index'])
    out['out_of_order_frame_indexes'] = out_of_order[:50]
    out['out_of_order_count'] = len(out_of_order)
    out['process_ids'] = sorted({m['pid'] for m in metas if 'pid' in m})
    out['frame_equals_index_plus'] = sorted({q['frame'] - q['frame_index'] for q in requests})[:10]
    out['missing_responses'] = sum(1 for r in responses if r is None)

    # ---- coverage and timing --------------------------------------------- #
    indexes = sorted({q['frame_index'] for q in requests})
    span = (indexes[-1] + 1) if indexes else 0
    total = total_frames or span
    gaps = [b - a - 1 for a, b in zip(indexes, indexes[1:])]
    out['coverage'] = {
        'frames_in_sequence': total,
        'frames_answered': len(indexes),
        'frames_skipped': total - len(indexes),
        'share_answered': round(len(indexes) / total, 3) if total else None,
        'first_frame_index': indexes[0] if indexes else None,
        'last_frame_index': indexes[-1] if indexes else None,
        'gap_histogram': dict(sorted(Counter(gaps).items())),
        'note': ('every skipped frame is scored with no detections, so share_answered '
                 'is roughly a ceiling on the score'),
    }
    times = [m.get('received_at') for m in metas]
    if all(t is not None for t in times) and len(times) > 1:
        deltas = [(b - a) * 1000.0 for a, b in zip(times, times[1:])]
        out['timing'] = {
            'ms_between_answers': _stats(deltas),
            'run_seconds': round(times[-1] - times[0], 1),
            'note': ('answers come at most once per frame (333 ms); a median well above '
                     '333 ms means the round trip (upload + our processing) is too slow'),
        }
    else:
        out['timing'] = {'note': 'no arrival timestamps (recording made before the recorder change)'}

    # ---- camera ------------------------------------------------------------ #
    levels = Counter(q['view']['resolution_level'] for q in requests)
    out['camera'] = {
        'views_by_level': {str(k): v for k, v in sorted(levels.items())},
        'refused_move_feedbacks': sum(1 for q in requests if q.get('camera_command_feedback')),
        'refusal_reasons': Counter(
            re.sub(r'[\d.]+', '#', q['camera_command_feedback'].get('reason', ''))
            for q in requests if q.get('camera_command_feedback')
        ).most_common(5),
        'responses_without_camera_request': sum(1 for r in responses if r and not r.get('requested_view')),
    }

    # ---- what we answered -------------------------------------------------- #
    per_frame, per_class, confidences = [], Counter(), []
    sizes = defaultdict(list)       # class -> [(w, h)] source px
    empty_streak = longest_empty = 0
    for r in responses:
        annotations = (r or {}).get('annotations') or []
        per_frame.append(len(annotations))
        empty_streak = 0 if annotations else empty_streak + 1
        longest_empty = max(longest_empty, empty_streak)
        for a in annotations:
            per_class[a['object_id']] += 1
            confidences.append(float(a['confidence']))
            x1, y1, x2, y2 = a['bbox']
            sizes[a['object_id']].append(((x2 - x1) * SOURCE_WIDTH, (y2 - y1) * SOURCE_HEIGHT))
    out['answers'] = {
        'annotations_per_frame': _stats(per_frame),
        'frames_with_no_annotations': sum(1 for n in per_frame if n == 0),
        'longest_empty_streak': longest_empty,
        'annotations_by_class': dict(per_class.most_common()),
        'confidence': _stats(confidences),
        'per_frame_counts': per_frame,
    }
    out['answered_box_sizes'] = _size_table(sizes)
    return out


def _size_table(sizes):
    """Per class: longest side in source px and in view px at each level."""
    table = {}
    for name, whs in sorted(sizes.items()):
        longest = [max(w, h) for w, h in whs]
        shortest = [min(w, h) for w, h in whs]
        median_long = statistics.median(longest)
        table[name] = {
            'boxes': len(whs),
            'longest_side_source_px': _stats(longest),
            'shortest_side_source_px_median': round(statistics.median(shortest), 1),
            'longest_side_view_px_median': {f'L{lvl}': round(median_long / s, 1) for lvl, s in LEVEL_SCALE.items()},
        }
    return table


def detect_views(run: Path, records, weights, imgsz, conf):
    """Raw detector output on every recorded view (no tracker), sizes in view px."""
    import cv2
    from ultralytics import YOLO

    model = YOLO(str(weights))
    names = {int(k): v for k, v in model.names.items()}
    by_level = defaultdict(lambda: {'views': 0, 'detections': 0, 'empty_views': 0})
    view_sizes = defaultdict(lambda: defaultdict(list))   # level -> class -> longest view px
    confidences = defaultdict(list)
    for record in records:
        png = run / (Path(record['_file']).stem + '.png')
        image = cv2.imread(str(png), cv2.IMREAD_COLOR)
        if image is None:
            continue
        level = record['request']['view']['resolution_level']
        result = model.predict(image, imgsz=imgsz, conf=conf, iou=0.5, agnostic_nms=True, verbose=False)[0]
        by_level[level]['views'] += 1
        count = 0 if result.boxes is None else len(result.boxes)
        by_level[level]['detections'] += count
        by_level[level]['empty_views'] += count == 0
        if count:
            for box, cls, score in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()):
                name = names.get(int(cls), str(cls))
                view_sizes[level][name].append(max(box[2] - box[0], box[3] - box[1]))
                confidences[level].append(float(score))
    return {
        'weights': str(weights),
        'imgsz': imgsz,
        'min_conf': conf,
        'by_level': {
            f'L{lvl}': {
                **counts,
                'detections_per_view': round(counts['detections'] / max(counts['views'], 1), 2),
                'confidence': _stats(confidences[lvl]),
                'longest_side_view_px_by_class': {
                    name: _stats(values) for name, values in sorted(view_sizes[lvl].items())
                },
                'longest_side_view_px_all': _stats([v for vs in view_sizes[lvl].values() for v in vs]),
            }
            for lvl, counts in sorted(by_level.items())
        },
        'note': ('YOLO P3 (stride 8) gets unreliable below roughly 8-10 view px; '
                 'many detections there would argue for a bigger input or a P2 head'),
    }


_FRAME_LINE = re.compile(
    r"frame (?P<frame>\d+) idx (?P<idx>\d+) L(?P<level>\d): (?P<dets>\d+) dets, (?P<tracks>\d+) tracks, "
    r"(?P<answers>\d+) answers.*?ms=(?P<ms>\{.*\})"
)


def parse_log(path: Path):
    totals, stage = [], defaultdict(list)
    dets_by_level = defaultdict(list)
    kinds = Counter()
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        match = _FRAME_LINE.search(line)
        if match:
            try:
                ms = json.loads(match['ms'].replace("'", '"'))
            except ValueError:
                ms = {}
            if 'total' in ms:
                totals.append(ms['total'])
            for name, value in ms.items():
                stage[name].append(value)
            dets_by_level[int(match['level'])].append(int(match['dets']))
            continue
        for marker, label in (
            ('!!! REPEATED REQUEST', 'repeated request'),
            ('!!! OUT-OF-ORDER', 'out-of-order request'),
            ('started over at frame_index 0', 'sequence restart'),
            ('restarted, resetting state', 'state wipe (old code)'),
            ('New sequence', 'new sequence while holding another'),
            ('Dropped state', 'sequence state evicted'),
            ('!!! DETECTOR FAILED', 'detector failing every frame'),
            ('!!! DETECTOR FOUND NOTHING', 'detector finding nothing'),
            ('Detector failed on frame', 'detector exception'),
            ('Camera command from frame', 'camera command refused'),
            ('Pipeline failed on frame', 'pipeline exception'),
            ('Traceback', 'traceback'),
            ('Refusing to start', 'refused to start'),
            ('Warmup done', 'server start'),
        ):
            if marker in line:
                kinds[label] += 1
    return {
        'pipeline_ms_total': _stats(totals),
        'pipeline_ms_by_stage_median': {k: round(statistics.median(v), 1) for k, v in stage.items() if v},
        'raw_detections_per_view_by_level': {f'L{k}': _stats(v) for k, v in sorted(dets_by_level.items())},
        'event_counts': dict(kinds.most_common()),
    }


def contact_sheet(run: Path, records, path: Path, count=24, tile=(320, 180)):
    import cv2
    import numpy as np

    if not records:
        return None
    step = max(1, len(records) // count)
    picked = records[::step][:count]
    cols = 6
    rows = math.ceil(len(picked) / cols)
    sheet = np.zeros((rows * tile[1], cols * tile[0], 3), dtype=np.uint8)
    for i, record in enumerate(picked):
        image = cv2.imread(str(run / (Path(record['_file']).stem + '.png')), cv2.IMREAD_COLOR)
        if image is None:
            continue
        small = cv2.resize(image, tile, interpolation=cv2.INTER_AREA)
        q = record['request']
        label = f"i{q['frame_index']} L{q['view']['resolution_level']} n{len((record.get('response') or {}).get('annotations') or [])}"
        cv2.putText(small, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(small, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        r, c = divmod(i, cols)
        sheet[r * tile[1]:(r + 1) * tile[1], c * tile[0]:(c + 1) * tile[0]] = small
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return path.name


def report_text(s):
    c, a = s['coverage'], s['answers']
    lines = [
        f"sequence(s): {', '.join(s['sequence_ids'])}",
        f"requests recorded: {s['requests_recorded']}",
        '',
        '== Coverage (skipped frames score zero) ==',
        f"answered {c['frames_answered']} of {c['frames_in_sequence']} frames "
        f"({(c['share_answered'] or 0) * 100:.0f}%), skipped {c['frames_skipped']}",
        f"gaps between answered frames (frames skipped: times): {c['gap_histogram']}",
        f"timing: {_fmt(s['timing'].get('ms_between_answers'), ' ms')}" if 'ms_between_answers' in s['timing']
        else f"timing: {s['timing']['note']}",
        '',
        '== Protocol health ==',
        f"repeated request_ids: {s['repeated_request_ids']}   out-of-order frame_index: {s['out_of_order_count']}",
        f"server process ids: {s['process_ids'] or 'n/a (old recording)'}"
        + ('   <-- SEVERAL PROCESSES ANSWERED THIS RUN' if len(s['process_ids']) > 1 else ''),
        f"frame - frame_index offsets seen: {s['frame_equals_index_plus']}",
        f"views by level: {s['camera']['views_by_level']}   refused camera moves: "
        f"{s['camera']['refused_move_feedbacks']} {s['camera']['refusal_reasons']}",
        '',
        '== What we answered ==',
        f"annotations per frame: {_fmt(a['annotations_per_frame'])}",
        f"frames with no annotations: {a['frames_with_no_annotations']}   longest empty streak: {a['longest_empty_streak']}",
        f"confidence: {_fmt(a['confidence'])}",
        f"by class: {a['annotations_by_class']}",
        '',
        '== Answered box sizes (longest side; view px = what the model sees at that level) ==',
    ]
    for name, row in s['answered_box_sizes'].items():
        v = row['longest_side_view_px_median']
        lines.append(f"  {name:16s} {row['boxes']:5d} boxes  source px median {row['longest_side_source_px']['median']:7.1f}"
                     f"   view px L0 {v['L0']:6.1f}  L1 {v['L1']:6.1f}  L2 {v['L2']:6.1f}")
    if 'detector_on_views' in s:
        lines += ['', '== Detector on recorded views (no tracker) ==']
        for lvl, row in s['detector_on_views']['by_level'].items():
            lines.append(f"  {lvl}: {row['views']} views, {row['detections_per_view']} detections/view, "
                         f"{row['empty_views']} empty; object size {_fmt(row['longest_side_view_px_all'], ' px')}")
    if 'server_log' in s:
        log = s['server_log']
        lines += ['', '== Server log ==',
                  f"pipeline ms per frame: {_fmt(log['pipeline_ms_total'], ' ms')}",
                  f"median ms by stage: {log['pipeline_ms_by_stage_median']}",
                  f"events: {log['event_counts']}"]
    return '\n'.join(lines) + '\n'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('path', nargs='?', default=str(ROOT / 'recordings'),
                        help='A recordings/<sequence_id> folder, or recordings/ (newest run is used)')
    parser.add_argument('--out', default='run_summary', help='Output folder')
    parser.add_argument('--total-frames', type=int, default=0,
                        help='Frames in the sequence (validation: 249). Default: last frame_index + 1')
    parser.add_argument('--log', help='Server log to parse as well (api.py stdout)')
    parser.add_argument('--detect', action='store_true', help='Run the detector on every recorded view')
    parser.add_argument('--weights', default=str(ROOT / 'weights' / 'detector.pt'))
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--conf', type=float, default=0.10)
    parser.add_argument('--no-sheet', action='store_true', help='Skip the contact sheet image')
    args = parser.parse_args()

    run = find_run(Path(args.path))
    records = load_records(run)
    if not records:
        print(f'No recorded requests in {run}', file=sys.stderr)
        return 1
    summary = {'run_folder': run.name, **summarize(records, args.total_frames)}
    if args.detect:
        summary['detector_on_views'] = detect_views(run, records, args.weights, args.imgsz, args.conf)
    if args.log:
        summary['server_log'] = parse_log(Path(args.log))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.no_sheet:
        summary['contact_sheet'] = contact_sheet(run, records, out / 'contact_sheet.jpg')
    (out / 'summary.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    text = report_text(summary)
    (out / 'summary.txt').write_text(text, encoding='utf-8')
    print(text)
    size_kb = sum(p.stat().st_size for p in out.iterdir()) / 1024
    print(f'Wrote {out}/ ({size_kb:.0f} KB). Send that folder, not recordings/.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
