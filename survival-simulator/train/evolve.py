"""
Evolutionary search over the hive policy's parameters.

A self-contained (mu, lambda) evolution strategy with rank selection and per-parameter
step-size adaptation. No extra dependencies - swapping in `cma` is straightforward if you
want it, but for ~35 loosely-coupled parameters a plain ES converges fine and keeps the
submission environment free of training-only packages.

Two things matter more than the choice of optimiser here:

* Budget, traded against truncation bias. A full 3000 s episode is minutes of wall clock,
  so `--max-time` truncates. But truncation is not neutral: trees die at age 50-100 while
  the tree spawn rate halves every 300 s, so food is abundant early and scarce late. A
  short episode therefore rewards camping one tree and punishes the cost of looking for
  another - which is exactly how an earlier run drove `w_explore` to zero. Train long
  enough to include the scarce regime, or the search will evolve the foraging behaviour
  back out.

* Overfitting. Validation and evaluation use different seeds, so the seed set is rotated
  every generation and fitness is downside-aware (mean minus a multiple of the spread).

    python train/evolve.py --resume checkpoints/forage_start.json --generations 20
    python train/evolve.py --resume checkpoints/best.json --generations 10 --max-time 3000
"""

import argparse
import json
import os
import random
import statistics
import sys
import time
from typing import List, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.controllers.params import Params, BOUNDS   # noqa: E402
from train.evaluate import evaluate, fitness              # noqa: E402

NAMES = Params.names()


def _init_sigma(scale: float) -> List[float]:
    return [scale * (BOUNDS[n][1] - BOUNDS[n][0]) for n in NAMES]


def _mutate(center: Sequence[float], sigma: Sequence[float], rng: random.Random) -> List[float]:
    return Params.clip_vector([c + rng.gauss(0.0, s) for c, s in zip(center, sigma)])


def _seeds_for_generation(gen: int, n: int, rng: random.Random) -> List[int]:
    """Fresh seeds every generation so the search cannot memorise a map."""
    return [rng.randint(1, 2 ** 31 - 1) for _ in range(n)]


def evolve(generations: int, pop_size: int, elites: int, n_seeds: int,
           max_time: float, sigma_scale: float, risk_aversion: float,
           out_dir: str, resume: str = None, seed: int = 0,
           workers: int = None) -> Params:
    rng = random.Random(seed)
    os.makedirs(out_dir, exist_ok=True)

    center = (Params.load(resume) if resume else Params()).to_vector()
    sigma = _init_sigma(sigma_scale)

    best_vec, best_fit = list(center), float("-inf")
    history = []

    for gen in range(generations):
        t0 = time.time()
        seeds = _seeds_for_generation(gen, n_seeds, rng)

        # the incumbent is re-evaluated on this generation's seeds so it has to keep
        # earning its place rather than coasting on a lucky draw
        candidates = [list(center)] + [_mutate(center, sigma, rng) for _ in range(pop_size - 1)]

        scored = []
        for vec in candidates:
            results = evaluate(Params.from_vector(vec), seeds, max_time, workers)
            f = fitness(results, risk_aversion)
            mean_surv = statistics.fmean(r["survived"] for r in results)
            scored.append((f, vec, mean_surv))

        scored.sort(key=lambda x: -x[0])
        top = scored[:elites]

        center = [statistics.fmean(v[i] for _, v, _ in top) for i in range(len(NAMES))]
        spread = [statistics.pstdev([v[i] for _, v, _ in top]) or 1e-6 for i in range(len(NAMES))]
        sigma = [max(0.75 * s + 0.25 * sp, 0.02 * (BOUNDS[n][1] - BOUNDS[n][0]) * 0.25)
                 for s, sp, n in zip(sigma, spread, NAMES)]

        if scored[0][0] > best_fit:
            best_fit, best_vec = scored[0][0], list(scored[0][1])
            Params.from_vector(best_vec).save(os.path.join(out_dir, "best.json"))

        Params.from_vector(center).save(os.path.join(out_dir, "center.json"))
        rec = {
            "gen": gen,
            "best_fitness": scored[0][0],
            "best_survived": scored[0][2],
            "elite_fitness": statistics.fmean(f for f, _, _ in top),
            "all_time_best": best_fit,
            "seconds": time.time() - t0,
        }
        history.append(rec)
        with open(os.path.join(out_dir, "history.json"), "w") as fh:
            json.dump(history, fh, indent=2)

        print(f"gen {gen:3d}  best_fit {scored[0][0]:8.1f}  survived {scored[0][2]:7.1f}s  "
              f"elite_avg {rec['elite_fitness']:8.1f}  all_time {best_fit:8.1f}  "
              f"({rec['seconds']:.0f}s)")

    return Params.from_vector(best_vec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=25)
    ap.add_argument("--pop", type=int, default=16, help="lambda")
    ap.add_argument("--elites", type=int, default=4, help="mu")
    ap.add_argument("--seeds", type=int, default=3, help="episodes per candidate")
    ap.add_argument("--max-time", type=float, default=2500.0,
                    help="truncate episodes; below ~2000 the search selects against foraging")
    ap.add_argument("--sigma", type=float, default=0.12, help="initial step as a fraction of range")
    ap.add_argument("--risk-aversion", type=float, default=0.5)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    print(f"evolving {len(NAMES)} parameters | pop {args.pop} | {args.seeds} seeds/candidate "
          f"| episodes truncated at {args.max_time:.0f}s")
    best = evolve(args.generations, args.pop, args.elites, args.seeds, args.max_time,
                  args.sigma, args.risk_aversion, args.out, args.resume,
                  args.rng_seed, args.workers)
    print("\nbest parameters written to", os.path.join(args.out, "best.json"))
    for n in NAMES:
        print(f"  {n:22s} {getattr(best, n):9.3f}")


if __name__ == "__main__":
    main()
