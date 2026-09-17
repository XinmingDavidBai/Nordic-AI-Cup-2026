"""Replay a scene through the pipeline IN-PROCESS, with pictures.

Same crops, payloads, camera rules and scorer as local_evaluator.py (it imports
them), but it calls ``pipeline.predictor.predict`` directly, so you can set
breakpoints, and it can draw every frame:

    python debug_replay.py --detector gt --viz          # tracking + policy with a perfect detector
    python debug_replay.py --detector yolo --viz        # your trained model
    python debug_replay.py --detector gt --policy hold_l0
    python debug_replay.py --detector yolo --realtime   # drop frames using measured latency
    python debug_replay.py --detector yolo --realtime --extra-latency-ms 150

Pictures go to debug_out/frame_XXXXXX.jpg (full frame at 1920x1080):
    thin green   ground truth
    coloured     our answers (class + confidence)
    yellow       region the camera showed this frame
    cyan dashed  view requested for next frame
    magenta dots raw detections this frame

Anything in pipeline/config.py can also be overridden through the environment.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scene', default=None)
    parser.add_argument('--detector', default=None, help='auto | yolo | gt | edges | none')
    parser.add_argument('--weights', default=None, help='Path to YOLO weights')
    parser.add_argument('--policy', default=None, help='greedy | hold_l0 | sweep_l1')
    parser.add_argument('--realtime', action='store_true', help='Skip frames according to measured processing time')
    parser.add_argument('--extra-latency-ms', type=float, default=30.0,
                        help='Added to measured time in --realtime (HTTP + JSON + base64 overhead)')
    parser.add_argument('--viz', action='store_true', help='Write annotated frames')
    parser.add_argument('--out', default='debug_out')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--dump-json', default=None, help='Write per-frame debug info to this file')
    args = parser.parse_args()

    # Config is read at import time, so set overrides before importing the pipeline.
    if args.detector:
        os.environ['DETECTOR_BACKEND'] = args.detector
    if args.weights:
        os.environ['DETECTOR_WEIGHTS'] = args.weights
    if args.policy:
        os.environ['POLICY_MODE'] = args.policy
    os.environ.setdefault('LOG_TIMINGS', '0')

    import logging

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    import cv2

    from dtos import DroneFlybyPredictRequestDto, DroneFlybyPredictResponseDto
    from local_evaluator import (
        FRAME_INTERVAL_SECONDS,
        Camera,
        CameraRejection,
        build_request,
        render_view,
        score,
    )
    from pipeline import predictor
    from utils import (
        DEFAULT_SCENE,
        draw_boxes,
        frame_numbers,
        global_bbox_to_source,
        load_annotations,
        load_frame,
        source_region_for_view,
        validate_response,
    )

    scene = args.scene or DEFAULT_SCENE
    frames = frame_numbers(scene)
    predictor.reset()
    predictor.warmup()

    out_dir = Path(args.out)
    if args.viz:
        out_dir.mkdir(parents=True, exist_ok=True)

    camera = Camera()
    feedback = None
    predictions = {}
    levels = Counter()
    refused = 0
    skipped = 0
    latencies = []
    dump = []
    virtual_clock = 0.0
    frame_index = 0

    while frame_index < len(frames):
        frame = frames[frame_index]
        image = load_frame(frame, scene)
        payload = build_request(frame, frame_index, camera, render_view(image, camera), feedback)
        request = DroneFlybyPredictRequestDto.model_validate(payload)
        shown_region = camera.source_region
        levels[camera.resolution_level] += 1

        started = time.perf_counter()
        response = predictor.predict(request)
        elapsed_ms = (time.perf_counter() - started) * 1e3
        latencies.append(elapsed_ms)

        validate_response(response)
        DroneFlybyPredictResponseDto.model_validate(response.model_dump())  # same check as the evaluator
        predictions[frame] = [
            {
                'object_id': a.object_id,
                'bbox': global_bbox_to_source(a.bbox),
                'confidence': float(a.confidence),
            }
            for a in response.annotations
        ]

        info = dict(predictor.last_debug)
        info['view'] = (camera.resolution_level, camera.center_x, camera.center_y)
        info['answers'] = len(response.annotations)
        dump.append(info)
        if args.verbose or frame_index % 5 == 0:
            print(
                f'frame {frame:3d} idx {frame_index:3d} L{camera.resolution_level} '
                f'({camera.center_x:4d},{camera.center_y:4d}) dets {len(info.get("detections", [])):2d} '
                f'tracks {info.get("tracks", 0):2d} answers {len(response.annotations):2d} '
                f'ego {info.get("displacement")} ok={info.get("ego_ok")} '
                f'{elapsed_ms:6.0f} ms  next {info.get("policy", {}).get("level")}@{info.get("policy", {}).get("center")}'
            )

        requested = response.requested_view
        if args.viz:
            canvas = image.copy()
            ground_truth = load_annotations(frame, scene)
            canvas = draw_boxes(canvas, ground_truth, labels=False, thickness=2)
            canvas = draw_boxes(canvas, predictions[frame], labels=True, thickness=4)
            cv2.rectangle(canvas, tuple(shown_region[:2]), tuple(shown_region[2:]), (0, 255, 255), 8)
            if requested is not None:
                r = source_region_for_view(requested.resolution_level, requested.center_x, requested.center_y)
                _dashed_rect(cv2, canvas, r, (255, 255, 0), 6)
            for _, _, box in info.get('detections', []):
                cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                cv2.circle(canvas, (cx, cy), 10, (255, 0, 255), -1)
            text = (
                f'frame {frame} idx {frame_index} L{camera.resolution_level} '
                f'dets {len(info.get("detections", []))} tracks {info.get("tracks", 0)} '
                f'ego {info.get("displacement")} {elapsed_ms:.0f}ms'
            )
            cv2.putText(canvas, text, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 5, cv2.LINE_AA)
            canvas = cv2.resize(canvas, (1920, 1080), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_dir / f'frame_{frame:06d}.jpg'), canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])

        if requested is not None:
            try:
                camera.apply(requested.resolution_level, requested.center_x, requested.center_y)
                feedback = None
            except CameraRejection as exc:
                refused += 1
                print(f'frame {frame}: camera command refused: {exc}')
                feedback = {
                    'frame': frame,
                    'requested_view': requested.model_dump(),
                    'reason': str(exc),
                }

        if not args.realtime:
            frame_index += 1
            continue
        virtual_clock += (elapsed_ms + args.extra_latency_ms) / 1000.0
        next_index = max(frame_index + 1, int(virtual_clock / FRAME_INTERVAL_SECONDS))
        skipped += max(0, min(next_index, len(frames)) - (frame_index + 1))
        frame_index = next_index

    coco_map, ap_by_class = score(scene, predictions)
    ordered = sorted(latencies)
    print()
    print(f'frames answered   {len(predictions)} / {len(frames)} (skipped {skipped})')
    print(f'camera levels     {dict(sorted(levels.items()))}   refused {refused}')
    print(f'pipeline ms       mean {sum(ordered) / len(ordered):.0f} / median {ordered[len(ordered) // 2]:.0f} / max {ordered[-1]:.0f}')
    print('AP@0.50 by class')
    for name, value in sorted(ap_by_class.items(), key=lambda kv: -kv[1]):
        print(f'  {name:16s} {value:.3f}')
    print(f'COCO mAP@0.50: {coco_map:.3f}')
    if args.viz:
        print(f'annotated frames in {out_dir.resolve()}')
    if args.dump_json:
        with open(args.dump_json, 'w', encoding='utf-8') as handle:
            json.dump(dump, handle, indent=1, default=str)
    return 0


def _dashed_rect(cv2, canvas, region, colour, thickness, dash=60):
    x1, y1, x2, y2 = (int(c) for c in region)
    for (ax, ay, bx, by) in ((x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)):
        length = max(abs(bx - ax), abs(by - ay))
        for start in range(0, length, dash * 2):
            end = min(start + dash, length)
            t0, t1 = start / length, end / length
            p0 = (int(ax + (bx - ax) * t0), int(ay + (by - ay) * t0))
            p1 = (int(ax + (bx - ax) * t1), int(ay + (by - ay) * t1))
            cv2.line(canvas, p0, p1, colour, thickness)


if __name__ == '__main__':
    sys.exit(main())
