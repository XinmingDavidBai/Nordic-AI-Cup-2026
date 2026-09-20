"""
Headless episode runner for the hive policy.

Runs the simulator in-process (no HTTP) so training is not paying request overhead.
Supports parallel evaluation across seeds, episode truncation for cheap early
generations, and optional rendering to watch a run.

    python train/evaluate.py --seeds 1 2 3 --max-time 3000
    python train/evaluate.py --params best.json --render
"""

import argparse
import os
import random
import sys
import time
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# every worker process imports pygame; one banner per worker floods the training log
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from src.core import SimulationCore                       # noqa: E402
from src.utils.controllers.params import Params           # noqa: E402
from src.utils.controllers.hive_policy import HiveMind    # noqa: E402

EPISODE_SECONDS = 3000.0

# Bump when the stream layout in _instrument changes: cached episodes keyed on it.
CRN_VERSION = 1


def _instrument(env, seed: int, crn: bool) -> Dict[str, int]:
    """
    Count births and deaths by cause, and optionally split the simulator's randomness
    into independent streams ("common random numbers").

    The simulator draws everything from one stream: tree and fruit spawns, tree deaths,
    predator arrivals, offspring mutations and predator wandering. The last two depend on
    what the hive does, so two policies on the same seed desynchronise at their first
    different spawn and see different trees and predators from then on - a paired
    comparison then shares little more than the map. With crn, the world stream (trees,
    fruit, predator arrivals) is consumed identically whatever the policy does; offspring
    and each predator's wandering draw from streams of their own. Every draw keeps its
    distribution, so the episodes are statistically the same game, just better matched.
    """
    cls = type(env)
    counts = {"births": 0, "starved": 0, "starved_old": 0, "eaten": 0}

    world = env.rng
    agent_rng = predator_seeds = None
    if crn:
        streams = random.Random(f"crn-{seed}")       # str seeds hash with sha512: stable
        world = random.Random(streams.getrandbits(64))
        agent_rng = random.Random(streams.getrandbits(64))
        predator_seeds = random.Random(streams.getrandbits(64))
        env.rng = world

    def spawn_agent(*args, **kwargs):
        if agent_rng is not None:
            env.rng = agent_rng
        try:
            agent = cls.spawn_agent(env, *args, **kwargs)
        finally:
            env.rng = world
        if agent is not None:
            counts["births"] += 1
        return agent

    def spawn_predator(*args, **kwargs):
        predator = cls.spawn_predator(env, *args, **kwargs)
        if predator is not None and predator_seeds is not None:
            predator.rng = random.Random(predator_seeds.getrandbits(64))
        return predator

    def kill_agent(agent):
        if agent in env.agents:
            if agent.energy > 0:
                counts["eaten"] += 1
            elif agent.age > agent.max_age:
                counts["starved_old"] += 1
            else:
                counts["starved"] += 1
        return cls.kill_agent(env, agent)

    # instance attributes shadow the class methods, including the env's own self.x() calls
    env.spawn_agent = spawn_agent
    env.spawn_predator = spawn_predator
    env.kill_agent = kill_agent
    return counts


class _Act:
    """Attribute-access shim - the simulator reads action.move_distance etc."""
    __slots__ = ("agent_id", "move_distance", "move_direction", "turn_angle", "spawn_agent")

    def __init__(self, d: dict):
        self.agent_id = d["agent_id"]
        self.move_distance = d["move_distance"]
        self.move_direction = d["move_direction"]
        self.turn_angle = d["turn_angle"]
        self.spawn_agent = d["spawn_agent"]


def run_episode(params: Params, seed: int, max_time: float = EPISODE_SECONDS,
                render: bool = False, verbose: bool = False,
                crn: bool = False, speedup: float = 1.0) -> Dict[str, float]:
    """
    Run one episode and return its metrics.

    speedup > 1 runs the environment's decline faster, for quicker screening: the tree
    spawn rate halves every 300/speedup s and predators arrive speedup times as fast
    (their number grows as 0.01 * speedup * t instead of 0.01 * t). Everything on the
    agents' own scale - tree and agent lifespans, fruit growth and rot, energy - is
    unchanged, and the policy and all reported times see the real clock. It is not the
    same game, only a faster proxy: confirm anything it finds at speedup 1.
    """
    if not render:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    sim = SimulationCore(seed=seed)
    counts = _instrument(sim.env, seed, crn)
    hive = HiveMind(params, rng_seed=seed)

    screen = clock = None
    if render:
        import pygame
        pygame.init()
        info = pygame.display.Info()
        h = int(info.current_h * 0.9)
        w = int(h * sim.env_width / sim.env_height)
        screen = pygame.display.set_mode((w, h), pygame.SCALED)
        clock = pygame.time.Clock()

    actions: List = []
    peak_pop = 0
    eaten_events = 0
    last_score = 0.0
    policy_time = 0.0
    ticks = 0
    wall0 = time.time()

    while True:
        state = sim.step(actions)
        ticks += 1
        if speedup != 1.0:
            env = sim.env
            # env.time only feeds the tree-spawn decay and the predator spawn chance
            env.time += (speedup - 1.0) * sim.dt
            # the native draw now sees speedup * t; a second draw with (speedup - 1) times
            # its chance makes the arrival rate speedup^2 * t / P, i.e. P = 0.01 * speedup * t
            extra = ((1.0 / max(1, len(env.predators))) * sim.dt * env.time * 0.0001
                     * (speedup - 1.0))
            if extra > env.rng.random():
                env.spawn_predator()
        now = sim.env.time / speedup
        pop = state["num_agents"]
        peak_pop = max(peak_pop, pop)
        if state["score"] < last_score:
            eaten_events += 1
        last_score = state["score"]

        if pop == 0 or now > max_time:
            break

        t0 = time.perf_counter()
        decided = hive.decide(state["observations"], now)
        policy_time += time.perf_counter() - t0
        actions = [(d["agent_id"], _Act(d)) for d in decided]

        if render:
            import pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    max_time = 0.0
            sim.env.draw(screen)
            font = pygame.font.SysFont(None, 24)
            for i, line in enumerate((
                f'Score {state["score"]:.1f}',
                f'Agents {pop}   Predators {len(sim.env.predators)}',
                f'Trees {len(sim.env.trees)}   Fruit {len(sim.env.fruits)}',
                f'Time {now:.0f}s',
            )):
                screen.blit(font.render(line, True, (255, 255, 255)), (20, 20 + 22 * i))
            pygame.display.flip()
            clock.tick(60)

        if verbose and ticks % 1000 == 0:
            print(f"  t={now:6.0f}  score={state['score']:8.1f}  pop={pop:3d}  "
                  f"preds={len(sim.env.predators):2d}  trees={len(sim.env.trees):3d}")

    if render:
        import pygame
        pygame.quit()

    return {
        "seed": float(seed),
        "score": float(sim.env.score),
        "survived": float(now),
        "survived_full": 1.0 if now > max_time - 1.0 else 0.0,
        "peak_pop": float(peak_pop),
        "final_pop": float(len(sim.env.agents)),
        "eaten_events": float(eaten_events),
        "births": float(counts["births"]),
        "starved": float(counts["starved"]),
        "starved_old": float(counts["starved_old"]),
        "eaten": float(counts["eaten"]),
        "ms_per_tick": 1000.0 * policy_time / max(ticks, 1),
        "wall": time.time() - wall0,
    }


def _worker(job):
    params_vec, seed, max_time, *rest = job
    crn = bool(rest[0]) if rest else False
    speedup = float(rest[1]) if len(rest) > 1 else 1.0
    return run_episode(Params.from_vector(params_vec), seed, max_time, crn=crn,
                       speedup=speedup)


def _indexed_worker(job):
    """_worker for a batch of several candidates: carries the candidate index through."""
    idx, inner = job
    return idx, _worker(inner)


def evaluate(params: Params, seeds: Sequence[int], max_time: float = EPISODE_SECONDS,
             workers: Optional[int] = None, crn: bool = False) -> List[Dict[str, float]]:
    """Evaluate one parameter set across seeds, in parallel where it helps."""
    jobs = [(params.to_vector(), s, max_time, crn) for s in seeds]
    if workers is None:
        workers = min(len(jobs), max(1, (os.cpu_count() or 2) - 1))
    if workers <= 1 or len(jobs) == 1:
        return [_worker(j) for j in jobs]
    import multiprocessing as mp
    with mp.Pool(workers) as pool:
        return pool.map(_worker, jobs)


def fitness(results: Sequence[Dict[str, float]], risk_aversion: float = 0.5) -> float:
    """
    Extinction is catastrophic and the final evaluation averages three runs, so optimise
    a downside-aware statistic rather than the raw mean.
    """
    scores = [r["score"] for r in results]
    n = len(scores)
    mean = sum(scores) / n
    if n < 2:
        return mean
    var = sum((s - mean) ** 2 for s in scores) / (n - 1)
    return mean - risk_aversion * var ** 0.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=None, help="JSON file of Params (default: built-in)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--max-time", type=float, default=EPISODE_SECONDS)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--crn", action="store_true",
                    help="split the simulator's randomness into streams (as training does)")
    args = ap.parse_args()

    params = Params.load(args.params) if args.params else Params()

    if args.render:
        r = run_episode(params, args.seeds[0], args.max_time, render=True, verbose=True,
                        crn=args.crn)
        results = [r]
    else:
        results = evaluate(params, args.seeds, args.max_time, args.workers, args.crn)

    print(f"\n{'seed':>6} {'score':>9} {'survived':>9} {'peak':>5} {'final':>6} "
          f"{'eaten':>6} {'ms/tick':>8} {'wall':>7}")
    for r in sorted(results, key=lambda x: x["seed"]):
        print(f"{r['seed']:6.0f} {r['score']:9.1f} {r['survived']:9.1f} "
              f"{r['peak_pop']:5.0f} {r['final_pop']:6.0f} {r['eaten_events']:6.0f} "
              f"{r['ms_per_tick']:8.2f} {r['wall']:7.0f}s")
    scores = [r["score"] for r in results]
    print(f"\nmean score {sum(scores)/len(scores):.1f}   "
          f"worst {min(scores):.1f}   fitness {fitness(results):.1f}")
    print(f"full-length survivals: {sum(r['survived_full'] for r in results):.0f}/{len(results)}")


if __name__ == "__main__":
    main()
