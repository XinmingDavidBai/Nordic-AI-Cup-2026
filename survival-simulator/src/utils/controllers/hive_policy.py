"""
Hive policy - one stateful controller driving the whole species.

Design notes (these follow directly from the simulator's mechanics):

* `move_direction` is applied RELATIVE to the agent's heading and costs the same in any
  direction, so the movement action is really a 2-D vector. Movement is therefore decided
  by superposing weighted steering vectors rather than by picking a winning "desire".

* `turn_angle` only aims the vision cone and decides the predator charge gate. It is
  decided by a separate policy from movement. Facing a predator at range > 90 px keeps it
  in its pivot branch instead of its charge branch, and costs nothing extra to combine
  with walking away.

* Aging (`0.01 * age` per tick, unscaled by dt) is the hard cap on a lineage. It is not
  observable directly, but crossing `max_age` is detectable within one tick by comparing
  the actual energy drop against the drop we can predict exactly from our own command.

* Each agent keeps three remembered points in its own dead-reckoned frame - last tree,
  last predator, last river cell. That is deliberately not a full map; it covers most of
  the value (return to camp, remember a threat bearing, bias across a river) for very
  little code and latency. A full landmark map is the natural extension.

The controller is stateful across ticks and across episodes; `decide()` detects a new
episode from `sim_time` going backwards and resets itself.
"""

import math
import random
from typing import Dict, List, Optional, Tuple

from src.utils.controllers.params import Params

TWO_PI = 2.0 * math.pi
HALF_PI = math.pi / 2.0

# move_penalty per biome, mirrored from src/elements/biome.py. The status payload tells us
# which biome the agent is standing in, so dead reckoning can be exact in open terrain.
BIOME_MOVE = {
    "forest": 1.0,
    "grassland": 1.0,
    "swamp": 0.5,
    "desert": 0.8,
    "river": 0.3,
}

# Trait caps from Environment.spawn_agent, used to normalise the trait score.
TRAIT_CAPS = {
    "speed": 20.0,
    "sprint_speed": 40.0,
    "hearing_radius": 100.0,
    "vision_radius": 400.0,
    "cone_angle": HALF_PI,
    "max_energy": 1000.0,
}


def _wrap(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return (angle + math.pi) % TWO_PI - math.pi


def _obs_key(o: dict):
    """
    Canonical ordering for an observation list.

    The simulator collects nearby entities into Python sets, which iterate in object-id
    order, so the observation list arrives in a different order between processes. Since
    the steering vectors are accumulated by summation and floating-point addition is not
    associative, that ordering leaks ~1e-16 differences into the action - which a
    30,000-tick feedback loop amplifies into completely different episodes. Sorting here
    makes a given (params, seed) pair exactly reproducible.
    """
    if "distance" in o:
        return (0, o["type"], o["distance"], o["angle"])
    (sx, sy), (ex, ey) = o["coords"]
    return (1, o["type"], sx, sy, ex, ey)


def _closest_point_on_edge(edge) -> Tuple[float, float]:
    """Closest point on an agent-local edge segment to the agent (the origin)."""
    (x1, y1), (x2, y2) = edge
    dx, dy = x2 - x1, y2 - y1
    denom = dx * dx + dy * dy
    if denom <= 1e-9:
        return x1, y1
    t = -(x1 * dx + y1 * dy) / denom
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return x1 + t * dx, y1 + t * dy


class AgentMemory:
    """Per-agent state the controller carries between ticks."""

    __slots__ = (
        "aid", "x", "y", "heading", "is_old", "trait_score",
        "pending_dx", "pending_dy", "pending_turn", "expected_drop", "last_energy",
        "tree_x", "tree_y", "tree_t", "pred_x", "pred_y", "pred_t",
        "river_x", "river_y", "river_t", "explore_dir", "commit_until", "born_t",
    )

    def __init__(self, aid: int, x: float, y: float, heading: float, t: float,
                 rng: random.Random):
        self.aid = aid
        self.x = x
        self.y = y
        self.heading = heading
        self.is_old = False
        self.trait_score = 0.0
        self.pending_dx = 0.0
        self.pending_dy = 0.0
        self.pending_turn = 0.0
        self.expected_drop = 0.0
        self.last_energy: Optional[float] = None
        self.tree_x = self.tree_y = 0.0
        self.tree_t = -1e9
        self.pred_x = self.pred_y = 0.0
        self.pred_t = -1e9
        self.river_x = self.river_y = 0.0
        self.river_t = -1e9
        self.explore_dir = rng.uniform(0.0, TWO_PI)
        self.commit_until = -1.0
        self.born_t = t


class HiveMind:
    """
    Stateful controller for the whole species.

    Usage:
        hive = HiveMind(params)
        actions = hive.decide(agent_status_list, sim_time)

    `agent_status_list` is the list of per-agent status dicts (the same shape as
    `ObservationResponse`). Returns a list of plain dicts - not pydantic models - because
    the per-tick latency budget is roughly 20 ms and DTO validation is pure overhead in
    the hot loop.
    """

    def __init__(self, params: Optional[Params] = None, rng_seed: int = 0):
        self.p = params if params is not None else Params()
        self._rng_seed = rng_seed
        self.reset()

    # ------------------------------------------------------------------ episode
    def reset(self) -> None:
        self.mem: Dict[int, AgentMemory] = {}
        self.time = -1.0
        self.rng = random.Random(self._rng_seed)
        self._spawn_requests: List[int] = []

    def _maybe_new_episode(self, sim_time: float) -> None:
        """The evaluation server runs three simulations back to back on one process."""
        if sim_time + 1e-6 < self.time:
            self.reset()
        self.time = sim_time

    # ------------------------------------------------------------------ helpers
    def _target_pop(self) -> float:
        p = self.p
        raw = p.pop_target_a * (2.0 ** (-self.time / 600.0)) + p.pop_target_b
        return min(p.pop_max, max(p.pop_min, raw))

    def _trait_score(self, st: dict) -> float:
        p = self.p
        return (
            p.w_speed * (st["speed"] / TRAIT_CAPS["speed"])
            + p.w_sprint * (st["sprint_speed"] / TRAIT_CAPS["sprint_speed"])
            + p.w_hearing * (st["hearing_radius"] / TRAIT_CAPS["hearing_radius"])
            + p.w_vision * (st["vision_range"] / TRAIT_CAPS["vision_radius"])
            + p.w_cone * (st["vision_angle"] / TRAIT_CAPS["cone_angle"])
            + p.w_maxe * (st["max_energy"] / TRAIT_CAPS["max_energy"])
        )

    def _advance_memory(self, m: AgentMemory, st: dict) -> None:
        """Apply the displacement we commanded last tick, then check for aging onset."""
        m.x += m.pending_dx
        m.y += m.pending_dy
        m.heading = _wrap(m.heading + m.pending_turn)
        m.pending_dx = m.pending_dy = m.pending_turn = 0.0

        energy = st["energy"]
        if m.last_energy is not None and not m.is_old:
            drop = m.last_energy - energy
            # Eating masks the signal (energy goes up), so only a larger-than-predicted
            # drop is informative. The aging penalty is 0.01*age per tick - at age 60
            # that is 0.6, two orders of magnitude above the 0.01 passive drain.
            if drop > m.expected_drop + 0.005 * max(st["age"], 1.0):
                m.is_old = True
        m.last_energy = energy

        if st.get("biome") == "river":
            m.river_x, m.river_y, m.river_t = m.x, m.y, self.time

    # ------------------------------------------------------------------ main
    def decide(self, agent_status: List[dict], sim_time: float) -> List[dict]:
        self._maybe_new_episode(sim_time)
        p = self.p

        if not agent_status:
            self._spawn_requests = []
            return []

        # --- register newborns. We know which agents we asked to spawn last tick, so a
        # child can be seeded at its parent's pose (it appears within 10-30 px of it).
        alive = set()
        parents = [self.mem[a] for a in self._spawn_requests if a in self.mem]
        pi = 0
        for st in agent_status:
            aid = st["agent_id"]
            alive.add(aid)
            if aid not in self.mem:
                if pi < len(parents):
                    src = parents[pi]
                    pi += 1
                    x, y = src.x, src.y
                else:
                    x = y = 0.0
                # A newborn's heading is randomised by the simulator and unknowable, so
                # the frame is arbitrary until it turns. Position error is ~20 px.
                self.mem[aid] = AgentMemory(aid, x, y, self.rng.uniform(0.0, TWO_PI),
                                            sim_time, self.rng)
        for dead in [a for a in self.mem if a not in alive]:
            del self.mem[dead]
        self._spawn_requests = []

        # --- population-level context
        pop = len(agent_status)
        scores = []
        for st in agent_status:
            s = self._trait_score(st)
            self.mem[st["agent_id"]].trait_score = s
            scores.append(s)
        scores.sort()
        idx = int(p.trait_gate_pct * (len(scores) - 1))
        trait_cut = scores[idx]
        target_pop = self._target_pop()

        actions = []
        for st in agent_status:
            m = self.mem[st["agent_id"]]
            self._advance_memory(m, st)
            actions.append(self._decide_one(st, m, pop, target_pop, trait_cut))
        return actions

    # ------------------------------------------------------------------ per agent
    def _decide_one(self, st: dict, m: AgentMemory, pop: int,
                    target_pop: float, trait_cut: float) -> dict:
        p = self.p
        energy = st["energy"]
        max_e = st["max_energy"]
        speed = st["speed"]
        sprint = st["sprint_speed"]

        # ---- parse observations in one pass
        fruit = None
        tree = None
        pred = None
        mates: List[dict] = []
        edges: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        for o in sorted(st["observations"], key=_obs_key):
            kind = o["type"]
            if kind == "Fruit":
                if fruit is None or o["distance"] < fruit["distance"]:
                    fruit = o
            elif kind == "Predator":
                if pred is None or o["distance"] < pred["distance"]:
                    pred = o
            elif kind == "Tree":
                if tree is None or o["distance"] < tree["distance"]:
                    tree = o
            elif kind == "Agent":
                mates.append(o)
            elif kind == "Edge":
                edges.append(o["coords"])

        # ---- fold sightings into memory (agent-local -> dead-reckoned frame)
        if tree is not None:
            a = m.heading + tree["angle"]
            m.tree_x = m.x + tree["distance"] * math.cos(a)
            m.tree_y = m.y + tree["distance"] * math.sin(a)
            m.tree_t = self.time
        if pred is not None:
            a = m.heading + pred["angle"]
            m.pred_x = m.x + pred["distance"] * math.cos(a)
            m.pred_y = m.y + pred["distance"] * math.sin(a)
            m.pred_t = self.time

        # ---- threat bearing: live sighting, else a recent remembered one
        pred_dist: Optional[float] = None
        pred_ang: Optional[float] = None
        if pred is not None:
            pred_dist, pred_ang = pred["distance"], pred["angle"]
        elif self.time - m.pred_t < min(6.0, p.memory_seconds):
            dx, dy = m.pred_x - m.x, m.pred_y - m.y
            pred_dist = math.hypot(dx, dy)
            pred_ang = _wrap(math.atan2(dy, dx) - m.heading)

        threatened = pred_dist is not None and pred_dist < p.danger_radius
        # An agent past max_age is dying anyway. Once the population can afford it, its
        # remaining energy is worth more as a predator distraction than as a forager -
        # and being eaten cheap costs almost nothing (-energy/100).
        decoy = m.is_old and pop >= p.decoy_pop and energy < p.decoy_energy

        # ---- steering: superpose weighted vectors in the agent-local frame
        vx = vy = 0.0
        hunger = 1.0 - min(1.0, energy / max(1.0, max_e * p.hunger_full))
        target_dist = None

        if fruit is not None and not threatened:
            if self.time >= m.commit_until:
                m.commit_until = self.time + p.commit_ticks * 0.1
            w = p.w_fruit * (0.35 + 0.65 * hunger)
            vx += w * math.cos(fruit["angle"])
            vy += w * math.sin(fruit["angle"])
            target_dist = fruit["distance"]

        # camp: hold station near the last known tree, which is where fruit appears
        if self.time - m.tree_t < p.memory_seconds:
            dx, dy = m.tree_x - m.x, m.tree_y - m.y
            d = math.hypot(dx, dy)
            if d > p.camp_radius:
                ang = _wrap(math.atan2(dy, dx) - m.heading)
                w = p.w_tree * min(1.0, (d - p.camp_radius) / 60.0)
                vx += w * math.cos(ang)
                vy += w * math.sin(ang)

        if pred_dist is not None and pred_ang is not None and not decoy:
            # inverse-square repulsion, away from the threat
            w = p.w_predator * (p.danger_radius / max(pred_dist, 12.0)) ** 2
            vx -= w * math.cos(pred_ang)
            vy -= w * math.sin(pred_ang)
        elif decoy and pred_dist is not None and pred_ang is not None:
            # lead it away from the swarm instead of fleeing it
            w = p.w_predator * 0.35
            vx += w * math.cos(pred_ang)
            vy += w * math.sin(pred_ang)

        for o in mates:
            d = o["distance"]
            if d < p.separation_radius:
                w = p.w_separation * (1.0 - d / p.separation_radius)
                vx -= w * math.cos(o["angle"])
                vy -= w * math.sin(o["angle"])

        for e in edges:
            ex, ey = _closest_point_on_edge(e)
            d = math.hypot(ex, ey)
            if d < p.edge_radius and d > 1e-6:
                w = p.w_edge * (1.0 - d / p.edge_radius)
                vx -= w * ex / d
                vy -= w * ey / d

        # crossing a river costs a predator a far larger share of its ~100 energy budget
        # than it costs us, so bias toward a remembered crossing when threatened
        if threatened and self.time - m.river_t < p.memory_seconds:
            dx, dy = m.river_x - m.x, m.river_y - m.y
            d = math.hypot(dx, dy)
            if d > 1e-6:
                ang = _wrap(math.atan2(dy, dx) - m.heading)
                vx += p.w_river * math.cos(ang)
                vy += p.w_river * math.sin(ang)

        # exploration: a slowly drifting persistent bearing keeps agents spreading out
        m.explore_dir = _wrap(m.explore_dir + self.rng.uniform(-0.15, 0.15))
        ang = _wrap(m.explore_dir - m.heading)
        w = p.w_explore * (1.0 if fruit is None and tree is None else 0.25)
        vx += w * math.cos(ang)
        vy += w * math.sin(ang)

        # ---- resolve into a move
        mag = math.hypot(vx, vy)
        if mag < 1e-6:
            move_dir = 0.0
            want_move = False
        else:
            move_dir = math.atan2(vy, vx)
            want_move = True

        camped = (self.time - m.tree_t < p.memory_seconds
                  and math.hypot(m.tree_x - m.x, m.tree_y - m.y) <= p.camp_radius)
        if (not threatened and fruit is None and camped
                and energy > max_e * p.idle_energy_frac):
            # standing still costs 0.1 energy/s against 5/s for walking
            want_move = False

        if not want_move:
            distance = 0.0
        elif threatened and pred_dist is not None and pred_dist < p.sprint_radius and not decoy:
            distance = sprint
        else:
            distance = speed
            if target_dist is not None and target_dist < distance:
                # don't overshoot a fruit - the eat radius is only ~10 px
                distance = max(target_dist, 0.0)

        # ---- heading: independent of travel direction
        if pred_ang is not None and not decoy:
            # hold the predator inside our forward half-plane so it stays in its pivot
            # branch instead of charging
            if abs(pred_ang) > p.face_margin:
                keep = max(0.0, p.face_margin - p.face_hysteresis)
                turn = pred_ang - math.copysign(keep, pred_ang)
            else:
                turn = 0.0
        elif want_move:
            # otherwise point roughly where we are walking so the cone scans ahead
            turn = max(-0.45, min(0.45, move_dir * 0.5))
        else:
            turn = 0.0

        spawn = self._want_spawn(st, m, pop, target_pop, trait_cut, threatened, energy)
        if spawn:
            self._spawn_requests.append(m.aid)

        self._stash_expectation(m, st, distance, move_dir, turn, speed, sprint, max_e,
                                energy, spawn)

        return {
            "agent_id": m.aid,
            "move_distance": distance,
            "move_direction": move_dir,
            "turn_angle": turn,
            "spawn_agent": spawn,
        }

    # ------------------------------------------------------------------ spawning
    def _want_spawn(self, st: dict, m: AgentMemory, pop: int, target_pop: float,
                    trait_cut: float, threatened: bool, energy: float) -> bool:
        p = self.p
        # Dying of old age: convert whatever is left into fresh age-0 agents. Still
        # ceilinged - an overshooting population starves together, and synchronised
        # starvation is how this policy goes extinct.
        if m.is_old and energy > p.spawn_energy_old and pop < target_pop * 1.5:
            return True
        # about to be chased: 100 energy in a child is 1 point of penalty we won't pay
        if threatened and energy > p.panic_spawn_energy:
            return True
        if energy < p.spawn_energy:
            return False
        if pop >= target_pop:
            return False
        # mutation drifts down ~0.44%/trait/generation, so breed from the upper tail -
        # unless the population is small enough that any child beats no child
        if pop > p.critical_pop and m.trait_score < trait_cut:
            return False
        return True

    # ------------------------------------------------------------------ odometry
    def _stash_expectation(self, m: AgentMemory, st: dict, distance: float,
                           move_dir: float, turn: float, speed: float, sprint: float,
                           max_e: float, energy: float, spawn: bool) -> None:
        """
        Predict this tick's displacement and energy drop exactly as the simulator will
        compute them, so next tick we can dead-reckon and detect the aging penalty.
        """
        eff = min(max(distance, 0.0), sprint)
        if energy < max_e / 5.0 and eff > speed:
            eff = speed

        if eff <= speed:
            move_cost = eff * 0.05
        else:
            move_cost = speed * 0.05 + (eff - speed) * 0.5
        turn_cost = min(math.pi, abs(turn)) / TWO_PI
        m.expected_drop = move_cost + turn_cost + 0.01  # + passive drain
        # Reproduction is charged in the same tick, after movement. Leaving it out of the
        # prediction makes every successful parent look like it just crossed max_age.
        if spawn and (energy - move_cost - turn_cost) > 100.0:
            m.expected_drop += 100.0

        eff *= BIOME_MOVE.get(st.get("biome"), 1.0)
        absolute = m.heading + move_dir
        m.pending_dx = eff * math.cos(absolute)
        m.pending_dy = eff * math.sin(absolute)
        m.pending_turn = turn
