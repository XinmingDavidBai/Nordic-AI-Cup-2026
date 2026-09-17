"""
Tunable parameters for the hive policy.

Every field here is a knob the evolutionary search is allowed to move. Defaults are
hand-set from the mechanics analysis (see the field manual) so the policy is already
functional before any training; the search refines from there.

BOUNDS keeps the search inside physically sensible ranges - without it the ES happily
proposes negative spawn thresholds and 900-pixel danger radii.
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
    pop_target_a: float = 12.0         # target(t) = a * 2**(-t/600) + b
    pop_target_b: float = 5.0
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

    "w_fruit":             (0.0, 6.0),
    "w_tree":              (0.0, 6.0),
    "w_predator":          (0.0, 12.0),
    "w_separation":        (0.0, 4.0),
    "w_edge":              (0.0, 4.0),
    "w_explore":           (0.0, 3.0),
    "w_river":             (0.0, 3.0),

    "separation_radius":   (10.0, 200.0),
    "edge_radius":         (6.0, 80.0),
    "camp_radius":         (8.0, 90.0),
    "hunger_full":         (0.30, 1.0),
    "commit_ticks":        (1.0, 60.0),
    "idle_energy_frac":    (0.0, 1.0),
    "memory_seconds":      (2.0, 90.0),
}
