"""Start api.py, run the organisers' local_evaluator.py against it over HTTP, stop.

Exactly what the competition does, on your machine, in one command:

    python run_local.py                              # offline, every frame
    python run_local.py --realtime                   # with the 3 fps clock
    python run_local.py --realtime --simulate-latency-ms 200
    python run_local.py --detector gt --verbose      # debug detector (local only)
    python run_local.py --policy sweep_l1

Anything after the known options is passed straight to local_evaluator.py.
Server logs go to debug_out/server.log.
"""

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--detector', help='auto | yolo | gt | edges | none')
    parser.add_argument('--weights', help='YOLO weights path')
    parser.add_argument('--policy', help='greedy | hold_l0 | sweep_l1')
    parser.add_argument('--port', type=int, default=9063,
                        help='Port for this test server (not 9053, so it never collides with a server you left running)')
    parser.add_argument('--startup-timeout', type=float, default=120.0)
    args, passthrough = parser.parse_known_args()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        if probe.connect_ex(('127.0.0.1', args.port)) == 0:
            print(f'Port {args.port} is already in use by another process. Pick another with --port.', file=sys.stderr)
            return 1

    # Not 'localhost': on Windows that tries IPv6 first and stalls ~2 s per new connection.
    base_url = f'http://127.0.0.1:{args.port}'
    env = os.environ.copy()
    env['PORT'] = str(args.port)
    if args.detector:
        env['DETECTOR_BACKEND'] = args.detector
        # Explicitly asked for a debug detector on this local test server, so let
        # api.py start with it (it refuses non-yolo detectors by default).
        if args.detector.lower() not in ('auto', 'yolo'):
            env['ALLOW_NON_YOLO_DETECTOR'] = '1'
    if args.weights:
        env['DETECTOR_WEIGHTS'] = str(Path(args.weights).resolve())
    if args.policy:
        env['POLICY_MODE'] = args.policy
    env.setdefault('PYTHONUNBUFFERED', '1')
    # api.py records by default (for validation runs); local test runs don't need it.
    env.setdefault('RECORD_DIR', '')

    log_dir = ROOT / 'debug_out'
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / 'server.log'
    with open(log_path, 'w', encoding='utf-8') as log:
        server = subprocess.Popen(
            [sys.executable, 'api.py'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
        )
        try:
            # The root only answers after startup, i.e. after model warmup.
            deadline = time.monotonic() + args.startup_timeout
            while True:
                if server.poll() is not None:
                    print(f'api.py exited early (code {server.returncode}); see {log_path}', file=sys.stderr)
                    return 1
                try:
                    requests.get(f'{base_url}/', timeout=1)
                    break
                except requests.RequestException:
                    if time.monotonic() > deadline:
                        print(f'api.py did not come up in {args.startup_timeout}s; see {log_path}', file=sys.stderr)
                        return 1
                    time.sleep(0.5)
            print(f'server up (logs: {log_path})')
            return subprocess.call(
                [sys.executable, 'local_evaluator.py', '--url', f'{base_url}/predict', *passthrough], cwd=ROOT
            )
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()


if __name__ == '__main__':
    sys.exit(main())
