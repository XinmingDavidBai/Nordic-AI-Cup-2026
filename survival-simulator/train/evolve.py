"""
Evolutionary search over the hive policy's parameters.

A self-contained (mu, lambda) evolution strategy: each generation's children mutate a
single parent drawn uniformly at random from the previous generation's elites - not the
elites' average - so a strong but unusual elite still gets explored on its own terms
instead of being blended away. The per-parameter step size still adapts to how much the
elites agree on that parameter. No extra dependencies - swapping in `cma` is
straightforward if you want it, but for ~35 loosely-coupled parameters a plain ES
converges fine and keeps the submission environment free of training-only packages.

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
import multiprocessing as mp
import os
import random
import statistics
import sys
import time
from typing import List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.controllers.params import Params, BOUNDS            # noqa: E402
from train.evaluate import fitness, _worker, _indexed_worker       # noqa: E402

NAMES = Params.names()

_MAX_DEFAULT_WORKERS = 12
# recycle each worker process after this many episodes, so slow leaks (pygame, numpy
# caches) cannot build up over a run that lasts hours
_TASKS_PER_WORKER = 25

# Generations of rescoring a contender needs before it can replace the champion. It is
# dropped as soon as it is not ahead on the seeds both have played.
_CONTEND_GENS = 2


def _init_sigma(scale: float) -> List[float]:
    return [scale * (BOUNDS[n][1] - BOUNDS[n][0]) for n in NAMES]


def _mutate(center: Sequence[float], sigma: Sequence[float], rng: random.Random) -> List[float]:
    return Params.clip_vector([c + rng.gauss(0.0, s) for c, s in zip(center, sigma)])


def _seeds_for_generation(gen: int, n: int, rng: random.Random) -> List[int]:
    """Fresh seeds every generation so the search cannot memorise a map."""
    return [rng.randint(1, 2 ** 31 - 1) for _ in range(n)]


def _default_workers() -> int:
    # Each worker is one single-threaded episode. Map generation briefly allocates
    # several ~150 MB arrays, and 16+ workers generating at once has run a 32 GB machine
    # out of memory, so the default is capped.
    return max(1, min((os.cpu_count() or 2) - 1, _MAX_DEFAULT_WORKERS))


def _evaluate_generation(candidates: Sequence[Sequence[float]], seeds: Sequence[int],
                         max_time: float, pool) -> List[List[dict]]:
    """
    Every candidate on every seed, as one batch: episode lengths vary several-fold
    (a hive can die out at 300 s or survive 3000 s), so feeding the whole generation to
    one pool keeps every worker busy instead of waiting on each candidate's slowest seed.
    Returns the results per candidate, in candidate order.
    """
    jobs = [(ci, (list(vec), s, max_time)) for ci, vec in enumerate(candidates) for s in seeds]
    results: List[List[dict]] = [[] for _ in candidates]
    if pool is None:
        for ci, job in jobs:
            results[ci].append(_worker(job))
    else:
        for ci, r in pool.imap_unordered(_indexed_worker, jobs, chunksize=1):
            results[ci].append(r)
    # completion order is arbitrary; seed order keeps the per-candidate lists comparable
    for rs in results:
        rs.sort(key=lambda r: r["seed"])
    return results


def evolve(generations: int, pop_size: int, elites: int, n_seeds: int,
           max_time: float, sigma_scale: float, risk_aversion: float,
           out_dir: str, resume: str = None, seed: int = 0,
           workers: int = None) -> Params:
    rng = random.Random(seed)
    os.makedirs(out_dir, exist_ok=True)

    # held to BOUNDS, like the policy does: a checkpoint from older, wider bounds
    center = Params.clip_vector((Params.load(resume) if resume else Params()).to_vector())
    sigma = _init_sigma(sigma_scale)

    # The champion is the best candidate seen so far. It is re-evaluated every generation
    # and always kept among the elites, so a generation can never lose it. Its fitness is
    # the mean over all its evaluations: episodes are noisy and not reproducible, so a
    # champion that only won on lucky seeds drifts down to its true level.
    # The starting parameters (--resume, or the defaults) are the first champion, so a run
    # never ends worse than where it started; they are the generation-0 centre, so they
    # get evaluated like everyone else.
    champ_vec: List[float] = list(center)
    champ_fits: List[float] = []
    # A generation's top score is the best of pop_size noisy draws, biased upward (the
    # winner's curse): with 1-5 seeds per candidate it beats nearly any fixed policy's
    # average, and the champion changed almost every generation. So the top challenger
    # only becomes a contender, re-evaluated alongside the champion on the following
    # generations' fresh seeds. Only those rescores count, and only against the
    # champion's scores on the same seeds: seed difficulty moves a whole generation
    # together (the elite average swung 939-1342 over one run), so comparing against the
    # champion's average over other seeds promoted whoever was rescored on an easy map.
    contender: Optional[List[float]] = None
    dropped: Optional[List[float]] = None
    contender_res: List[dict] = []      # its rescores, pooled over generations
    champ_res: List[dict] = []          # the champion's episodes on the same seeds
    contender_gens = 0
    history = []

    # Mutants are perturbations of a randomly chosen elite from the previous generation,
    # not of their average. There is no previous generation yet, so gen 0 falls back to
    # mutating the resume vector (or the built-in defaults) - the same as every generation
    # did before this change.
    elite_vecs: List[List[float]] = [list(center)]

    # One pool for the whole run: starting worker processes costs seconds each, and a
    # candidate-by-candidate pool left most workers idle behind each slow episode.
    # Workers are daemonic, so Ctrl+C still takes them down with the parent.
    if workers is None:
        workers = _default_workers()
    pool = mp.Pool(workers, maxtasksperchild=_TASKS_PER_WORKER) if workers > 1 else None

    for gen in range(generations):
        t0 = time.time()
        seeds = _seeds_for_generation(gen, n_seeds, rng)

        # the elites' average and the champion are re-evaluated on this generation's
        # seeds, so both have to keep earning their place; the rest of the population is
        # filled by mutating a uniformly-random elite parent, redrawn per child
        candidates = [list(center)]
        for keep in (champ_vec, contender):
            if keep is not None and keep not in candidates:
                candidates.append(list(keep))
        candidates += [_mutate(rng.choice(elite_vecs), sigma, rng)
                       for _ in range(pop_size - len(candidates))]

        evaluated = _evaluate_generation(candidates, seeds, max_time, pool)
        results_of = {tuple(v): rs for v, rs in zip(candidates, evaluated)}
        scored = []
        for vec, results in zip(candidates, evaluated):
            f = fitness(results, risk_aversion)
            mean_surv = statistics.fmean(r["survived"] for r in results)
            scored.append((f, vec, mean_surv))
        scored.sort(key=lambda x: -x[0])

        # update the champion: fold in its new evaluation, then score the contender
        # against it on the seeds both have now played
        champ_entry = next(e for e in scored if e[1] == champ_vec)
        champ_fits.append(champ_entry[0])
        champion_changed = False
        contender_score = contender_lead = None
        lead_gens = 0
        if contender is not None:
            ce = next(e for e in scored if e[1] == contender)
            contender_score = ce[0]
            contender_res += results_of[tuple(contender)]
            champ_res += results_of[tuple(champ_vec)]
            contender_gens += 1
            lead_gens = contender_gens
            contender_lead = (fitness(contender_res, risk_aversion)
                              - fitness(champ_res, risk_aversion))
            if contender_lead <= 0.0:
                dropped, contender = contender, None
            elif contender_gens >= _CONTEND_GENS:
                champ_vec, champ_fits, champ_entry = list(contender), [ce[0]], ce
                champion_changed = True
                contender = None
        if gen == 0 or champion_changed:
            Params.from_vector(champ_vec).save(os.path.join(out_dir, "best.json"))
        champ_fit = statistics.fmean(champ_fits)
        # this generation's best newcomer becomes the next contender, if it beat the
        # champion on this generation's seeds (and none is still being tested)
        new_contender_score = None
        if contender is None:
            challenger = next((e for e in scored
                               if e[1] != champ_vec and e[1] != dropped), None)
            if challenger is not None and challenger[0] > champ_entry[0]:
                contender = list(challenger[1])
                contender_res, champ_res, contender_gens = [], [], 0
                new_contender_score = challenger[0]

        # elites: the top of this generation, with the champion always among them. These
        # become next generation's pool of mutation parents (elite_vecs) - each child
        # picks one uniformly at random, rather than mutating their blended average.
        top = scored[:elites]
        if not any(v == champ_vec for _, v, _ in top):
            top = top[:elites - 1] + [champ_entry]
        elite_vecs = [v for _, v, _ in top]

        # center is no longer the mutation basis - it is evaluated each generation purely
        # as a "what if we blended the elites" candidate, and saved for inspection/resume
        center = [statistics.fmean(v[i] for _, v, _ in top) for i in range(len(NAMES))]
        spread = [statistics.pstdev([v[i] for _, v, _ in top]) or 1e-6 for i in range(len(NAMES))]
        sigma = [max(0.75 * s + 0.25 * sp, 0.02 * (BOUNDS[n][1] - BOUNDS[n][0]) * 0.25)
                 for s, sp, n in zip(sigma, spread, NAMES)]

        Params.from_vector(center).save(os.path.join(out_dir, "center.json"))
        rec = {
            "gen": gen,
            "best_fitness": scored[0][0],
            "best_survived": scored[0][2],
            "elite_fitness": statistics.fmean(f for f, _, _ in top),
            "champion_fitness": champ_fit,
            "champion_evaluations": len(champ_fits),
            "champion_changed": champion_changed,
            "contender_rescore": contender_score,
            "contender_lead": contender_lead,
            "contender_gens": lead_gens,
            "next_contender_score": new_contender_score,
            "seconds": time.time() - t0,
        }
        history.append(rec)
        with open(os.path.join(out_dir, "history.json"), "w") as fh:
            json.dump(history, fh, indent=2)

        rescore = ("" if contender_score is None else
                   f"  contender rescored {contender_score:8.1f} "
                   f"(lead {contender_lead:+7.1f} over {lead_gens} gen{'s' * (lead_gens > 1)})")
        print(f"gen {gen:3d}  best_fit {scored[0][0]:8.1f}  survived {scored[0][2]:7.1f}s  "
              f"elite_avg {rec['elite_fitness']:8.1f}  champion {champ_fit:8.1f} "
              f"({'new' if champion_changed else f'{len(champ_fits)} evals'}){rescore}  "
              f"({rec['seconds']:.0f}s)")

    if pool is not None:
        pool.close()
        pool.join()
    return Params.from_vector(champ_vec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=25)
    # One episode's score spreads ~100-230 points on its own, more than most differences
    # between parameter sets, so fewer candidates on more seeds beats the reverse. A run
    # at 24 x 1 seed selected mostly on luck: its generation winners dropped ~400 points
    # on average when rescored.
    ap.add_argument("--pop", type=int, default=12, help="lambda")
    ap.add_argument("--elites", type=int, default=4, help="mu")
    ap.add_argument("--seeds", type=int, default=4, help="episodes per candidate")
    ap.add_argument("--max-time", type=float, default=2500.0,
                    help="truncate episodes; below ~2000 the search selects against foraging")
    ap.add_argument("--sigma", type=float, default=0.12, help="initial step as a fraction of range")
    ap.add_argument("--risk-aversion", type=float, default=0.5)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None,
                    help=f"parallel episodes (default: CPU count - 1, at most {_MAX_DEFAULT_WORKERS}); "
                         "no longer limited by --seeds")
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
