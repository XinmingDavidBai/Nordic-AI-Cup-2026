"""
Tunable parameters for the reactive agent policy.

Every field here is a knob the evolutionary search is allowed to move. Defaults are
hand-set from the mechanics analysis (see the field manual) so the policy is already
functional before any training; the search refines from there.

BOUNDS keeps the search inside physically sensible ranges - without it the ES happily
proposes negative spawn thresholds and 900-pixel danger radii.

The policy is purely reactive: an agent decides from the observation list of the current
tick and nothing else. It keeps no map, no remembered landmark and no estimate of where
it is, so there is no parameter here for memory, localisation or coordination. The one
piece of shared information left is the population count, which the simulator hands us in
the status list rather than something the hive infers.

Measured dead ends, so they are not re-explored by hand: raising the population target
above ~15 costs score reliably (the hive overshoots the fruit supply and starves
together). Differences below about 150 points need 20+ paired seeds to resolve at all;
between-seed spread swamps anything smaller.
"""

from dataclasses import dataclass, fields, asdict
from typing import Dict, List, Tuple
import json


@dataclass
class Params:
    # ---- reproduction -------------------------------------------------------
    spawn_energy: float = 200.0        # normal spawn gate
    pop_target: float = 12.0           # agents the hive aims to keep alive
    pop_max: float = 35.0              # hard ceiling, even for a full parent
    # Eating is capped at max_energy, so fruit eaten while full is thrown away: a full
    # agent breeds even past the population target.
    spawn_full_frac: float = 0.9       # breed at this fraction of max_energy regardless of target (>1 off)
    # A parent that found a rich patch used to spawn every tick while it could pay 100 -
    # 4-5 children in half a second, all at one tree, where they starved.
    spawn_cooldown: float = 15.0       # seconds between one agent's spawns

    # ---- predator response --------------------------------------------------
    danger_radius: float = 130.0       # start evading (predators charge within 90 px)
    sprint_radius: float = 95.0        # burn sprint energy inside this
    # A predator only chases what it perceives: within its 60 degree cone out to 250 px,
    # or within 60 px in any direction. Agents ignore predators that cannot perceive them
    # (they still face them - that is cheap).
    pred_aware_margin: float = 0.6     # radians of predator turn allowed for, on top of its cone

    # ---- steering weights ---------------------------------------------------
    w_fruit: float = 1.6
    w_tree: float = 0.7
    w_predator: float = 3.2
    w_separation: float = 0.5
    w_edge: float = 0.7
    w_explore_hungry: float = 1.2      # exploration weight once foraging has failed

    # ---- steering geometry --------------------------------------------------
    separation_radius: float = 55.0
    explore_separation_radius: float = 120.0  # spread out while sweeping, not while at a tree
    edge_radius: float = 22.0
    camp_radius: float = 30.0          # this close to a visible tree counts as being at it
    hunger_full: float = 0.80          # fraction of max_energy above which fruit stops pulling
    starve_frac: float = 0.25          # below this fraction of max_energy, eat anything
    idle_energy_frac: float = 0.55     # at a tree and above this -> stand still (0.1 energy/s)
    move_threshold: float = 0.15       # a steering sum weaker than this is not worth walking for

    # ---- foraging -----------------------------------------------------------
    # Tree spawn rate is halved every 300 s while trees die at age 50-100, so the map goes
    # from ~120 trees to a handful over a full episode. Standing where the food used to be
    # is fatal late; sweeping costs energy, so it is only worth paying once nothing has
    # been in view for a while.
    forage_patience: float = 25.0      # seconds with no fruit/tree in view before exploring (0 off)

    # ---- look-around --------------------------------------------------------
    # The cone is ~60 degrees, so an agent walking or idling one way is blind to fruit a
    # few pixels off to the side. A full turn in cone-width steps costs ~1 energy in
    # total - five ticks of walking - and it does not interrupt movement.
    scan_interval: float = 5.0         # seconds between 360 degree scans (0 off)
    scan_alert_interval: float = 2.0   # ... while this agent has seen a predator in the last 20 s

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
            data = json.load(fh)
        # checkpoints from older versions carry parameters that no longer exist
        known = set(cls.names())
        return cls(**{k: v for k, v in data.items() if k in known})


BOUNDS: Dict[str, Tuple[float, float]] = {
    "spawn_energy":        (105.0, 420.0),
    "pop_target":          (1.0, 40.0),
    "pop_max":             (5.0, 80.0),
    "spawn_full_frac":     (0.5, 1.01),
    "spawn_cooldown":      (3.0, 60.0),

    # a predator charges anything within 90 px (1.5 x its hearing) whichever way it faces:
    # an agent must start evading before that and sprint inside it (training found 121/93)
    "danger_radius":       (100.0, 400.0),
    "sprint_radius":       (80.0, 160.0),
    "pred_aware_margin":   (0.0, 1.6),

    "w_fruit":             (0.0, 6.0),
    "w_tree":              (0.0, 6.0),
    "w_predator":          (0.0, 12.0),
    "w_separation":        (0.0, 4.0),
    "w_edge":              (0.0, 4.0),
    "w_explore_hungry":    (0.0, 3.0),

    "separation_radius":   (10.0, 200.0),
    "explore_separation_radius": (10.0, 300.0),
    "edge_radius":         (6.0, 80.0),
    "camp_radius":         (8.0, 90.0),
    "hunger_full":         (0.30, 1.0),
    "starve_frac":         (0.0, 0.8),
    "idle_energy_frac":    (0.0, 1.0),
    "move_threshold":      (0.0, 1.0),

    "forage_patience":     (0.0, 120.0),

    "scan_interval":       (0.0, 15.0),
    "scan_alert_interval": (0.5, 10.0),
}
