"""
Fixed-seed comparison of parameter sets.

Training scores are measured on fresh seeds every generation, and seed difficulty moves
every candidate together - one unchanged parameter set scored 804-1303 over five of them -
so they cannot tell whether a run improved anything. This plays every parameter set on the
same fixed seeds and compares them seed by seed, which cancels how hard each map was.

Two seed sets: "val" is what es.py gates on, so after a long run it flatters the result
slightly; "test" is never used in training and gives the honest number.

Episodes are cached in checkpoints/eval_cache.jsonl, keyed on the parameters, seed,
episode length, random-stream mode and the simulator/policy source, so repeating a
comparison only plays what is new, and editing the policy invalidates old entries.

    python train/validate.py checkpoints/train_paired/best.json checkpoints/best.json defaults
    python train/validate.py checkpoints/es/start.json checkpoints/es/best.json --set test --seeds 24
"""

import argparse
import glob
import hashlib
import json
import math
import multiprocessing as mp
import os
import statistics
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils.controllers.params import Params                     # noqa: E402
from train.evaluate import CRN_VERSION, EPISODE_SECONDS, _indexed_worker  # noqa: E402

CACHE_PATH = os.path.join(ROOT, "checkpoints", "eval_cache.jsonl")

# far from the small seeds used by hand in evaluate.py, and from each other
_SEED_BASE = {"val": 910_000, "test": 920_000}

# Map generation briefly allocates several ~150 MB arrays per worker; 19 workers has run
# a 32 GB machine out of memory.
_MAX_DEFAULT_WORKERS = 12
# recycle workers so slow leaks (pygame, numpy caches) cannot build up over hours
TASKS_PER_WORKER = 25


def seed_set(name: str, n: int, offset: int = 0) -> List[int]:
    return [_SEED_BASE[name] + offset + i for i in range(n)]


def default_workers() -> int:
    return max(1, min((os.cpu_count() or 2) - 1, _MAX_DEFAULT_WORKERS))


def _source_fingerprint() -> str:
    """Everything an episode's outcome depends on, apart from the parameter values."""
    files = sorted(glob.glob(os.path.join(ROOT, "src", "**", "*.py"), recursive=True))
    h = hashlib.sha1()
    for path in files:
        # params.py only holds defaults and bounds; the cache key has every value itself
        if os.path.basename(path) == "params.py":
            continue
        with open(path, "rb") as fh:
            h.update(os.path.relpath(path, ROOT).encode())
            h.update(fh.read())
    return h.hexdigest()


class EpisodeCache:
    """Append-only record of played episodes. path=None keeps it in memory only."""

    def __init__(self, path: Optional[str] = CACHE_PATH):
        self.path = path
        self.fingerprint = _source_fingerprint()
        self._data: Dict[str, dict] = {}
        if path and os.path.exists(path):
            with open(path) as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except ValueError:      # a line cut short by a crash
                        continue
                    self._data[rec["key"]] = rec["result"]

    def key(self, vec: Sequence[float], seed: int, max_time: float, crn: bool,
            speedup: float = 1.0) -> str:
        named = {n: round(float(v), 9) for n, v in zip(Params.names(), vec)}
        fields = [self.fingerprint, CRN_VERSION if crn else 0, named, int(seed), float(max_time)]
        if speedup != 1.0:
            fields.append(float(speedup))       # absent at 1, so older entries stay valid
        blob = json.dumps(fields, sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()

    def get(self, key: str) -> Optional[dict]:
        return self._data.get(key)

    def put(self, key: str, result: dict) -> None:
        self._data[key] = result
        if self.path:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "a") as fh:
                fh.write(json.dumps({"key": key, "result": result}) + "\n")


def run_batch(jobs: Sequence[Tuple[Sequence[float], int]], max_time: float, crn: bool,
              pool, cache: EpisodeCache,
              progress: Optional[Callable[[int, int], None]] = None,
              speedup: float = 1.0) -> List[dict]:
    """
    Play every (parameter vector, seed) job and return the results in job order. Cached
    episodes are not replayed and duplicate jobs are played once. The whole batch goes to
    the pool at once: episode lengths vary several-fold, so this keeps workers busy.
    """
    results: List[Optional[dict]] = [None] * len(jobs)
    pending: Dict[str, Tuple[tuple, List[int]]] = {}
    for i, (vec, seed) in enumerate(jobs):
        k = cache.key(vec, seed, max_time, crn, speedup)
        hit = cache.get(k)
        if hit is not None:
            results[i] = hit
        elif k in pending:
            pending[k][1].append(i)
        else:
            pending[k] = ((list(vec), int(seed), max_time, crn, speedup), [i])

    keys = list(pending)
    work = [(ki, pending[k][0]) for ki, k in enumerate(keys)]
    outcomes = (map(_indexed_worker, work) if pool is None
                else pool.imap_unordered(_indexed_worker, work, chunksize=1))
    for done, (ki, r) in enumerate(outcomes, 1):
        cache.put(keys[ki], r)
        for i in pending[keys[ki]][1]:
            results[i] = r
        if progress:
            progress(done, len(work))
    return results  # type: ignore[return-value]


def summarize(scores: Sequence[float]) -> Dict[str, float]:
    n = len(scores)
    return {
        "n": n,
        "mean": statistics.fmean(scores),
        "se": statistics.stdev(scores) / math.sqrt(n) if n > 1 else math.inf,
        "median": statistics.median(scores),
        "worst": min(scores),
    }


def paired(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """a minus b, seed by seed. z = mean / standard error; |z| < 2 is not a reliable gap."""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    mean = statistics.fmean(d)
    se = statistics.stdev(d) / math.sqrt(n) if n > 1 else math.inf
    if se == 0:
        z = 0.0 if mean == 0 else math.copysign(math.inf, mean)
    else:
        z = mean / se
    try:
        corr = statistics.correlation(a, b)
    except (statistics.StatisticsError, ValueError):   # constant input or n < 2
        corr = math.nan
    return {"n": n, "mean": mean, "se": se, "z": z, "corr": corr,
            "wins": sum(x > 0 for x in d), "losses": sum(x < 0 for x in d)}


def _load(spec: str) -> Params:
    return Params() if spec == "defaults" else Params.load(spec)


def _progress_line(done: int, total: int) -> None:
    print(f"\r  {done}/{total} episodes", end="" if done < total else "\n", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("params", nargs="+",
                    help="parameter JSON files, or 'defaults'; the first is the baseline")
    ap.add_argument("--set", choices=sorted(_SEED_BASE), default="val")
    ap.add_argument("--seeds", type=int, default=16, help="how many seeds of the set")
    ap.add_argument("--offset", type=int, default=0,
                    help="skip this many seeds of the set - fresh seeds to confirm a pick on")
    ap.add_argument("--max-time", type=float, default=EPISODE_SECONDS)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--no-crn", action="store_true",
                    help="play with the simulator's single random stream")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--speedup", type=float, default=1.0,
                    help="run the environment's decline this much faster (screening only)")
    args = ap.parse_args()

    crn = not args.no_crn
    seeds = seed_set(args.set, args.seeds, args.offset)
    vecs = [Params.clip_vector(_load(p).to_vector()) for p in args.params]
    cache = EpisodeCache(None if args.no_cache else CACHE_PATH)
    workers = args.workers or default_workers()

    print(f"{len(args.params)} parameter sets x {len(seeds)} {args.set} seeds "
          f"({seeds[0]}..{seeds[-1]}) | episodes up to {args.max_time:.0f}s | "
          f"common random numbers {'on' if crn else 'off'}"
          + (f" | environment decline x{args.speedup:g}" if args.speedup != 1.0 else ""))
    jobs = [(v, s) for v in vecs for s in seeds]
    pool = mp.Pool(workers, maxtasksperchild=TASKS_PER_WORKER) if workers > 1 else None
    try:
        flat = run_batch(jobs, args.max_time, crn, pool, cache, _progress_line,
                         args.speedup)
    finally:
        if pool is not None:
            pool.terminate()
    per = [flat[i * len(seeds):(i + 1) * len(seeds)] for i in range(len(vecs))]

    width = max(12, *(len(p) for p in args.params))
    print(f"\n{'':{width}s} {'mean':>8s} {'+-se':>6s} {'median':>7s} {'worst':>7s} "
          f"{'full':>6s} {'births':>7s} {'starved':>8s} {'old':>5s} {'eaten':>6s}")
    for name, rs in zip(args.params, per):
        s = summarize([r["score"] for r in rs])
        full = f"{sum(r['survived_full'] for r in rs):.0f}/{len(rs)}"
        avg = lambda k: statistics.fmean(r.get(k, math.nan) for r in rs)   # noqa: E731
        print(f"{name:{width}s} {s['mean']:8.1f} {s['se']:6.1f} {s['median']:7.1f} "
              f"{s['worst']:7.1f} {full:>6s} {avg('births'):7.1f} "
              f"{avg('starved'):8.1f} {avg('starved_old'):5.1f} {avg('eaten'):6.1f}")
    print("(births and deaths are per-episode averages; 'old' = starved past max_age)")

    if len(per) > 1:
        base = [r["score"] for r in per[0]]
        print(f"\npaired against {args.params[0]}, seed by seed:")
        for name, rs in zip(args.params[1:], per[1:]):
            p = paired([r["score"] for r in rs], base)
            print(f"{name:{width}s} {p['mean']:+8.1f} +- {p['se']:5.1f}  z {p['z']:+5.2f}  "
                  f"better on {p['wins']}/{p['n']}  corr {p['corr']:.2f}")
        print("|z| below ~2 is within noise. corr is how much the seed decides the score; "
              "the higher it is, the more pairing helps.")


if __name__ == "__main__":
    main()
