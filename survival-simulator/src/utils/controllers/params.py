"""
Tunable parameters for the hive policy.

Every field here is a knob the evolutionary search is allowed to move. Defaults are
hand-set from the mechanics analysis (see the field manual) so the policy is already
functional before any training; the search refines from there.

BOUNDS keeps the search inside physically sensible ranges - without it the ES happily
proposes negative spawn thresholds and 900-pixel danger radii.

Measured dead ends, so they are not re-explored by hand: raising the population target
above ~15 costs score reliably (the hive overshoots the fruit supply and starves
together), and re-weighting the trait gate moves neither the score nor the traits it
selects on - paired over 20 seeds it came out at -3 +- 58 points. Differences below about
150 points need 20+ paired seeds to resolve at all; between-seed spread swamps anything
smaller.
"""

from dataclasses import dataclass, fields, asdict
from typing import Dict, List, Tuple
import json
import math


@dataclass
class Params:
    # ---- reproduction -------------------------------------------------------
    spawn_energy: float = 200.0        # normal spawn gate
    spawn_energy_old: float = 115.0    # past max_age: dump energy into children
    panic_spawn_energy: float = 145.0  # spawn while threatened (de-risks the -E/100 penalty)
    pop_target_a: float = 12.0         # target(t) = a * 2**(-t/tau) + b
    pop_target_b: float = 5.0
    pop_target_tau: float = 600.0      # half-life of the early breeding allowance
    pop_min: float = 3.0
    pop_max: float = 35.0
    trait_gate_pct: float = 0.35       # only breed from agents above this population quantile
    critical_pop: float = 5.0          # below this, ignore the trait gate entirely

    # ---- trait scoring (drives the breeding gate) ---------------------------
    w_speed: float = 1.0
    w_hearing: float = 1.4
    w_vision: float = 0.8
    w_cone: float = 0.4
    w_sprint: float = 0.5
    w_maxe: float = 0.5

    # ---- predator response --------------------------------------------------
    danger_radius: float = 130.0       # start evading
    sprint_radius: float = 60.0        # burn sprint energy inside this
    face_margin: float = 1.15          # keep predator within this of heading (< pi/2)
    face_hysteresis: float = 0.30      # overshoot the correction to avoid per-tick turning
    decoy_pop: float = 8.0             # old agents decoy only if population is this healthy
    decoy_energy: float = 60.0         # a decoy wants to be cheap to lose
    # The relay cuts predator kills ~16%; on its own that bought no score, because runs
    # end on starvation. Kept on because it also carries food sightings, and because
    # predator spawn rate grows with time - it may start paying on longer episodes.
    w_relay: float = 1.0               # trust in a mate's predator sighting (0 disables)
    w_cover: float = 0.0               # how far to bend a flee toward broken line of sight

    # ---- steering weights ---------------------------------------------------
    w_fruit: float = 1.6
    w_tree: float = 0.7
    w_predator: float = 3.2
    w_separation: float = 0.5
    w_edge: float = 0.7
    w_explore: float = 0.35
    w_river: float = 0.25

    # ---- steering geometry --------------------------------------------------
    separation_radius: float = 55.0
    edge_radius: float = 22.0
    camp_radius: float = 30.0
    hunger_full: float = 0.80          # fraction of max_energy above which fruit stops pulling
    commit_ticks: float = 10.0         # stay locked on a fruit target this long
    idle_energy_frac: float = 0.55     # camped and above this -> stand still (0.1 energy/s)
    memory_seconds: float = 25.0       # how long a remembered landmark stays trusted

    # ---- fruit ripeness -----------------------------------------------------
    # Fruit spawns at 20 energy, grows 2/s to a cap of 60, and rots at age 100. Waiting for
    # ripeness measures worse than it sounds: agents already eat fruit at ~41 energy
    # average because it ages before anyone finds it, so a 20 s wait buys 8% more per
    # fruit while losing 36% of them to rot. Off by default; ripeness is not observable,
    # so when enabled it is timed from first sighting.
    fruit_ripe_wait: float = 0.0       # seconds to let a fruit ripen (0 disables waiting)
    fruit_match_radius: float = 12.0   # tolerance for re-identifying a fruit across ticks
    starve_frac: float = 0.25          # below this fraction of max_energy, eat anything
    mate_memory_seconds: float = 3.0   # how long a mate's pose stays usable for relaying

    # ---- foraging coverage --------------------------------------------------
    # Tree spawn rate is halved every 300 s while trees die at age 50-100, so the map goes
    # from ~120 trees to a handful over a full episode. Camping is correct early and fatal
    # late: an agent whose tree dies has no drive at all once its 20 s memory expires.
    # Worth +421 points over 5 paired seeds (t=3.7, 5/5). tree_giveup carries most of it:
    # dropping it alone collapses the gain to noise, because an agent that will not leave
    # a spent tree has nowhere to be once the tree dies.
    forage_patience: float = 25.0      # seconds with no fruit/tree before exploring (0 off)
    w_explore_hungry: float = 1.2      # exploration weight once foraging has failed
    explore_separation_radius: float = 120.0  # spread out while sweeping, not while camped
    tree_giveup: float = 40.0          # seconds of a barren camp before moving on (0 off)
    tree_memory_seconds: float = 90.0  # remembered trees outlive the generic landmark memory
    w_food_relay: float = 1.0          # hand tree sightings to mates as well (0 off)

    # ------------------------------------------------------------------ utils
    def to_vector(self) -> List[float]:
        return [float(getattr(self, f.name)) for f in fields(self)]

    @classmethod
    def names(cls) -> List[str]:
        return [f.name for f in fields(cls)]

    @classmethod
    def from_vector(cls, vec) -> "Params":
        return cls(**{n: float(v) for n, v in zip(cls.names(), vec)})

    @classmethod
    def clip_vector(cls, vec) -> List[float]:
        out = []
        for n, v in zip(cls.names(), vec):
            lo, hi = BOUNDS[n]
            out.append(min(hi, max(lo, float(v))))
        return out

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "Params":
        with open(path) as fh:
            return cls(**json.load(fh))


BOUNDS: Dict[str, Tuple[float, float]] = {
    "spawn_energy":        (105.0, 420.0),
    "spawn_energy_old":    (101.0, 260.0),
    "panic_spawn_energy":  (101.0, 320.0),
    "pop_target_a":        (0.0, 60.0),
    "pop_target_b":        (1.0, 30.0),
    # pinned: an early breeding burst measured worse on 9 of 10 held-out seeds
    "pop_target_tau":      (600.0, 600.0),
    "pop_min":             (1.0, 12.0),
    "pop_max":             (5.0, 80.0),
    "trait_gate_pct":      (0.0, 0.9),
    "critical_pop":        (1.0, 20.0),

    "w_speed":             (0.0, 3.0),
    "w_hearing":           (0.0, 3.0),
    "w_vision":            (0.0, 3.0),
    "w_cone":              (0.0, 3.0),
    "w_sprint":            (0.0, 3.0),
    "w_maxe":              (0.0, 3.0),

    "danger_radius":       (45.0, 400.0),
    "sprint_radius":       (20.0, 160.0),
    "face_margin":         (0.30, math.pi / 2 - 0.05),
    "face_hysteresis":     (0.0, 0.9),
    "decoy_pop":           (2.0, 40.0),
    "decoy_energy":        (5.0, 250.0),
    "w_relay":             (0.0, 1.0),
    "w_cover":             (0.0, 1.0),

    "w_fruit":             (0.0, 6.0),
    "w_tree":              (0.0, 6.0),
    "w_predator":          (0.0, 12.0),
    "w_separation":        (0.0, 4.0),
    "w_edge":              (0.0, 4.0),
    # pinned: superseded by w_explore_hungry, and the search had already driven it to 0
    "w_explore":           (0.0, 0.0),
    "w_river":             (0.0, 3.0),

    "separation_radius":   (10.0, 200.0),
    "edge_radius":         (6.0, 80.0),
    "camp_radius":         (8.0, 90.0),
    "hunger_full":         (0.30, 1.0),
    "commit_ticks":        (1.0, 60.0),
    "idle_energy_frac":    (0.0, 1.0),
    "memory_seconds":      (2.0, 90.0),

    # pinned: waiting for ripeness gains 8% per fruit but loses 36% of them to rot
    "fruit_ripe_wait":     (0.0, 0.0),
    "fruit_match_radius":  (4.0, 30.0),
    "starve_frac":         (0.0, 0.8),
    "mate_memory_seconds": (0.0, 10.0),

    "forage_patience":     (0.0, 120.0),
    "w_explore_hungry":    (0.0, 3.0),
    "explore_separation_radius": (10.0, 300.0),
    "tree_giveup":         (0.0, 120.0),
    "tree_memory_seconds": (10.0, 300.0),
    "w_food_relay":        (0.0, 1.0),
}
