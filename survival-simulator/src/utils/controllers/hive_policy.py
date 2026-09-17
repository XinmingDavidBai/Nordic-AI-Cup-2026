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

# candidate escape directions tested for broken line of sight
_COVER_SAMPLES = 12

# a tree is a big, sparse landmark; fruit spawns within ~3 tree radii of its trunk
_TREE_MATCH_RADIUS = 25.0
_TREE_FRUIT_RADIUS = 90.0

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


def _segments_cross(p1, p2, p3, p4) -> bool:
    """True if segment p1-p2 properly crosses segment p3-p4."""
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1, d2 = orient(p3, p4, p1), orient(p3, p4, p2)
    d3, d4 = orient(p1, p2, p3), orient(p1, p2, p4)
    return (d1 > 0.0) != (d2 > 0.0) and (d3 > 0.0) != (d4 > 0.0)


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
        "mates", "fruits", "trees", "food_t",
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
        # id -> (x, y, heading, t) in this agent's dead-reckoned frame
        self.mates: Dict[int, Tuple[float, float, float, float]] = {}
        # [x, y, first_seen, last_seen] per tracked fruit, same frame
        self.fruits: List[List[float]] = []
        # [x, y, last_seen, last_fruit_seen] per remembered tree, same frame
        self.trees: List[List[float]] = []
        self.food_t = t


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
        self._relay: Dict[int, Tuple[float, float]] = {}

    @staticmethod
    def _parse(st: dict) -> dict:
        """One sorted pass over an agent's observations (see _obs_key for why sorted)."""
        fruits: List[dict] = []
        mates: List[dict] = []
        edges: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        tree = pred = None
        for o in sorted(st["observations"], key=_obs_key):
            kind = o["type"]
            if kind == "Fruit":
                fruits.append(o)
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
        return {"fruits": fruits, "tree": tree, "pred": pred, "mates": mates, "edges": edges}

    def _update_mate_memory(self, m: AgentMemory, mates: List[dict]) -> None:
        """
        Remember where each mate was and which way it faced. The vision cone is only ~60
        degrees, so mates drop out of sight constantly; a few seconds of memory keeps a
        warning reaching agents the spotter can no longer see.
        """
        p = self.p
        if p.mate_memory_seconds <= 0.0:
            m.mates.clear()
            return
        for o in mates:
            aid = o.get("id")
            if aid is None:
                continue
            a = m.heading + o["angle"]
            bx = m.x + o["distance"] * math.cos(a)
            by = m.y + o["distance"] * math.sin(a)
            # rel_dir is the bearing from the mate back to us, in the mate's own frame
            heading_b = math.atan2(m.y - by, m.x - bx) - o["rel_dir"]
            m.mates[aid] = (bx, by, heading_b, self.time)
        stale = [k for k, v in m.mates.items() if self.time - v[3] > p.mate_memory_seconds]
        for k in stale:
            del m.mates[k]

    def _update_fruit_memory(self, m: AgentMemory, fruits: List[dict]) -> None:
        """Track first-sighting time per fruit; ripeness is not observable directly."""
        p = self.p
        if p.fruit_ripe_wait <= 0.0:
            return
        for o in fruits:
            a = m.heading + o["angle"]
            fx = m.x + o["distance"] * math.cos(a)
            fy = m.y + o["distance"] * math.sin(a)
            best, best_d = None, p.fruit_match_radius
            for e in m.fruits:
                d = math.hypot(e[0] - fx, e[1] - fy)
                if d < best_d:
                    best, best_d = e, d
            if best is None:
                m.fruits.append([fx, fy, self.time, self.time])
            else:
                best[0], best[1], best[3] = fx, fy, self.time
        m.fruits = [e for e in m.fruits
                    if self.time - e[3] < p.memory_seconds][-12:]

    def _merge_tree(self, m: AgentMemory, tx: float, ty: float, seen: bool) -> None:
        best, best_d = None, _TREE_MATCH_RADIUS
        for e in m.trees:
            d = math.hypot(e[0] - tx, e[1] - ty)
            if d < best_d:
                best, best_d = e, d
        if best is None:
            # a relayed tree starts with its productivity clock running, so it is not
            # written off before the recipient has had a chance to walk over and look
            m.trees.append([tx, ty, self.time, self.time])
        elif seen:
            best[0], best[1], best[2] = tx, ty, self.time

    def _update_tree_memory(self, m: AgentMemory, tree, fruits: List[dict]) -> None:
        """Keep several trees, and track which of them still produce."""
        p = self.p
        if tree is not None:
            a = m.heading + tree["angle"]
            self._merge_tree(m,
                             m.x + tree["distance"] * math.cos(a),
                             m.y + tree["distance"] * math.sin(a), True)
        for o in fruits:
            a = m.heading + o["angle"]
            fx = m.x + o["distance"] * math.cos(a)
            fy = m.y + o["distance"] * math.sin(a)
            if not m.trees:
                break
            near = min(m.trees, key=lambda e: math.hypot(e[0] - fx, e[1] - fy))
            if math.hypot(near[0] - fx, near[1] - fy) < _TREE_FRUIT_RADIUS:
                near[3] = self.time
        m.trees = [e for e in m.trees
                   if self.time - e[2] < p.tree_memory_seconds][-8:]

    def _pick_camp(self, m: AgentMemory):
        """Nearest remembered tree that still looks productive, else None -> go explore."""
        p = self.p
        if not m.trees:
            return None
        cands = m.trees
        if p.tree_giveup > 0.0:
            cands = [e for e in m.trees if self.time - e[3] < p.tree_giveup]
            if not cands:
                return None
        return min(cands, key=lambda e: math.hypot(e[0] - m.x, e[1] - m.y))

    def _fruit_age(self, m: AgentMemory, o: dict) -> float:
        a = m.heading + o["angle"]
        fx = m.x + o["distance"] * math.cos(a)
        fy = m.y + o["distance"] * math.sin(a)
        for e in m.fruits:
            if math.hypot(e[0] - fx, e[1] - fy) < self.p.fruit_match_radius:
                return self.time - e[2]
        return 0.0

    def _pick_fruit(self, m: AgentMemory, fruits: List[dict], starving: bool,
                    threatened: bool, speed: float) -> Optional[dict]:
        """Prefer ripe fruit; break the wait when starving, or to grab one before fleeing."""
        p = self.p
        if not fruits:
            return None
        if p.fruit_ripe_wait <= 0.0:
            # disabled: nearest fruit, and a flee ignores food entirely
            return None if threatened else min(fruits, key=lambda o: o["distance"])
        if threatened:
            # about to be driven off the patch - take one already within a single step,
            # which is a flee that collects rather than a detour toward the predator
            near = [o for o in fruits if o["distance"] <= speed]
            return min(near, key=lambda o: o["distance"]) if near else None
        if starving:
            return min(fruits, key=lambda o: o["distance"])
        ripe = [o for o in fruits if self._fruit_age(m, o) >= p.fruit_ripe_wait]
        return min(ripe, key=lambda o: o["distance"]) if ripe else None

    def _build_relay(self, parsed: Dict[int, dict]) -> Dict[int, Tuple[float, float]]:
        """
        Second-hand predator bearings, keyed by the recipient's agent_id.

        Each spotter works purely in its own dead-reckoned frame: it places the predator
        and its remembered mates in that frame, and the mate's stored heading (recovered
        from rel_dir when it was last seen) converts the difference into the mate's own
        body frame. The frames cancel, so no shared map is needed. Worth the trig because
        a predator only charges an agent facing away from it - an agent told where to look
        holds it in its harmless pivot branch.
        """
        relay: Dict[int, Tuple[float, float]] = {}
        for aid, pr in parsed.items():
            pred = pr["pred"]
            if pred is None:
                continue
            m = self.mem[aid]
            a = m.heading + pred["angle"]
            px = m.x + pred["distance"] * math.cos(a)
            py = m.y + pred["distance"] * math.sin(a)
            for bid, (bx, by, heading_b, _t) in m.mates.items():
                vx, vy = px - bx, py - by
                d = math.hypot(vx, vy)
                prev = relay.get(bid)
                if prev is None or d < prev[0]:
                    relay[bid] = (d, _wrap(math.atan2(vy, vx) - heading_b))
        return relay

    def _share_trees(self, parsed: Dict[int, dict]) -> None:
        """
        Pool tree discoveries across the hive using the same frame conversion as the
        predator relay. This is the part of a shared world map that actually pays: food
        is what runs out, and a scout's find is useless if only the scout knows it.
        """
        for aid, pr in parsed.items():
            tree = pr["tree"]
            if tree is None:
                continue
            m = self.mem[aid]
            a = m.heading + tree["angle"]
            tx = m.x + tree["distance"] * math.cos(a)
            ty = m.y + tree["distance"] * math.sin(a)
            for bid, (bx, by, heading_b, _t) in m.mates.items():
                other = self.mem.get(bid)
                if other is None:
                    continue
                vx, vy = tx - bx, ty - by
                d = math.hypot(vx, vy)
                ang = _wrap(math.atan2(vy, vx) - heading_b)
                a2 = other.heading + ang
                self._merge_tree(other, other.x + d * math.cos(a2),
                                 other.y + d * math.sin(a2), False)

    def _maybe_new_episode(self, sim_time: float) -> None:
        """The evaluation server runs three simulations back to back on one process."""
        if sim_time + 1e-6 < self.time:
            self.reset()
        self.time = sim_time

    # ------------------------------------------------------------------ helpers
    def _target_pop(self) -> float:
        p = self.p
        raw = p.pop_target_a * (2.0 ** (-self.time / max(p.pop_target_tau, 1.0))) + p.pop_target_b
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

        # Memory has to advance for every agent before any relaying, because a spotter
        # places its mates in its own dead-reckoned frame.
        parsed: Dict[int, dict] = {}
        for st in agent_status:
            m = self.mem[st["agent_id"]]
            self._advance_memory(m, st)
            pr = self._parse(st)
            parsed[st["agent_id"]] = pr
            if p.w_relay > 0.0 or p.w_food_relay > 0.0:
                self._update_mate_memory(m, pr["mates"])
            self._update_fruit_memory(m, pr["fruits"])
            if p.forage_patience > 0.0 or p.tree_giveup > 0.0 or p.w_food_relay > 0.0:
                self._update_tree_memory(m, pr["tree"], pr["fruits"])
            # Seeing a tree counts as foraging success, not just fruit. Gaps between fruit
            # spawns are normal, and triggering a sweep on them abandons good ground -
            # measured ~260 points worse over 5 paired seeds.
            if pr["tree"] is not None or pr["fruits"]:
                m.food_t = self.time

        self._relay = self._build_relay(parsed) if p.w_relay > 0.0 else {}
        if p.w_food_relay > 0.0:
            self._share_trees(parsed)

        actions = []
        for st in agent_status:
            aid = st["agent_id"]
            actions.append(self._decide_one(st, self.mem[aid], parsed[aid],
                                            pop, target_pop, trait_cut))
        return actions

    # ------------------------------------------------------------------ per agent
    def _decide_one(self, st: dict, m: AgentMemory, pr: dict, pop: int,
                    target_pop: float, trait_cut: float) -> dict:
        p = self.p
        energy = st["energy"]
        max_e = st["max_energy"]
        speed = st["speed"]
        sprint = st["sprint_speed"]

        tree, pred = pr["tree"], pr["pred"]
        mates, edges = pr["mates"], pr["edges"]

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
        elif m.aid in self._relay:
            pred_dist, pred_ang = self._relay[m.aid]

        threatened = pred_dist is not None and pred_dist < p.danger_radius
        # An agent past max_age is dying anyway. Once the population can afford it, its
        # remaining energy is worth more as a predator distraction than as a forager -
        # and being eaten cheap costs almost nothing (-energy/100).
        decoy = m.is_old and pop >= p.decoy_pop and energy < p.decoy_energy

        starving = energy < max_e * p.starve_frac
        fruit = self._pick_fruit(m, pr["fruits"], starving, threatened, speed)

        use_trees = p.tree_giveup > 0.0 or p.forage_patience > 0.0
        camp = self._pick_camp(m) if use_trees else None
        # sweeping is triggered by observed scarcity rather than by a clock: it costs
        # energy, and it is only worth paying once the local patch has stopped producing
        exploring = (p.forage_patience > 0.0 and fruit is None and not threatened
                     and self.time - m.food_t > p.forage_patience)

        # ---- steering: superpose weighted vectors in the agent-local frame
        vx = vy = 0.0
        hunger = 1.0 - min(1.0, energy / max(1.0, max_e * p.hunger_full))
        target_dist = None

        # _pick_fruit only returns a target while threatened if it is within one step, so
        # this stays a flee that happens to collect rather than a detour into the predator
        if fruit is not None:
            if self.time >= m.commit_until:
                m.commit_until = self.time + p.commit_ticks * 0.1
            w = p.w_fruit * (0.35 + 0.65 * hunger)
            vx += w * math.cos(fruit["angle"])
            vy += w * math.sin(fruit["angle"])
            target_dist = fruit["distance"]

        # camp: hold station near a known tree, which is where fruit appears
        if camp is not None:
            camp_x, camp_y = camp[0], camp[1]
        elif not use_trees and self.time - m.tree_t < p.memory_seconds:
            camp_x, camp_y = m.tree_x, m.tree_y
        else:
            camp_x = camp_y = None

        if camp_x is not None and not exploring:
            dx, dy = camp_x - m.x, camp_y - m.y
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

        # while sweeping, push apart hard enough that the hive covers ground instead of
        # re-searching the same patch together - dispersion needs no shared frame
        sep_r = p.explore_separation_radius if exploring else p.separation_radius
        for o in mates:
            d = o["distance"]
            if d < sep_r:
                w = p.w_separation * (1.0 - d / sep_r)
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
        if exploring:
            w = p.w_explore_hungry
        else:
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

        camped = (camp_x is not None
                  and math.hypot(camp_x - m.x, camp_y - m.y) <= p.camp_radius)
        if (not threatened and fruit is None and camped and not exploring
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

        # ---- cover: bend the escape toward a step that puts an obstacle between us and
        # the predator. Predator vision is 250 against 50 of hearing, so breaking line of
        # sight cuts its detection range fivefold and costs no sprint energy.
        if (p.w_cover > 0.0 and threatened and want_move and distance > 1e-6
                and edges and pred_dist is not None and pred_ang is not None and not decoy):
            pred_pt = (pred_dist * math.cos(pred_ang), pred_dist * math.sin(pred_ang))
            best_dev = best_dir = None
            for k in range(_COVER_SAMPLES):
                a = _wrap(move_dir + (k - _COVER_SAMPLES // 2) * (TWO_PI / _COVER_SAMPLES))
                cand = (distance * math.cos(a), distance * math.sin(a))
                if any(_segments_cross((0.0, 0.0), cand, e[0], e[1]) for e in edges):
                    continue  # that step walks into the obstacle instead of behind it
                if any(_segments_cross(cand, pred_pt, e[0], e[1]) for e in edges):
                    dev = abs(_wrap(a - move_dir))
                    if best_dev is None or dev < best_dev:
                        best_dev, best_dir = dev, a
            if best_dir is not None:
                move_dir = _wrap(move_dir + p.w_cover * _wrap(best_dir - move_dir))

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
