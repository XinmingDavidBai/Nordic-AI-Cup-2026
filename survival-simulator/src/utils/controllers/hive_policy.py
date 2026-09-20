"""
Reactive policy - one controller object driving the whole species, but no shared map.

Each agent decides from the observation list of the current tick and nothing else. It
does not know where it is, where it has been, or what any other agent can see. The only
state carried between ticks is a handful of per-agent timers (see AgentState), and the
only cross-agent fact is the population count, which the simulator hands us in the status
list rather than something the hive works out.

Design notes (these follow directly from the simulator's mechanics):

* `move_direction` is applied RELATIVE to the agent's heading and costs the same in any
  direction, so the movement action is really a 2-D vector. Movement is therefore decided
  by superposing weighted steering vectors rather than by picking a winning "desire".

* `turn_angle` only aims the vision cone and decides the predator charge gate. It is
  decided by a separate policy from movement. Facing a predator at range > 90 px keeps it
  in its pivot branch instead of its charge branch, and costs nothing extra to combine
  with walking away.

* Every bearing this policy stores is relative to the agent's own heading, and is rotated
  by the turn we command at the end of each tick. Turning is exact - the simulator applies
  our `turn_angle` unmodified - so a stored bearing never drifts, unlike a position, which
  obstacle deflection and wall clamping would ruin within seconds.

* Crowding. Two agents at one tree split its fruit, so when an agent sees another with a
  lower id, it is the one that leaves: it explores for _CONFLICT_SECONDS. Ids are the only
  usable key, since a mate observation carries `id` but not age, and both agents read the
  same comparison the same way, so exactly one of any pair gives way.

The controller is stateful across ticks and across episodes; `decide()` detects a new
episode from `sim_time` going backwards and resets itself.
"""

import math
import random
from typing import Dict, List, Optional, Tuple

from src.utils.controllers.params import Params

TWO_PI = 2.0 * math.pi
HALF_PI = math.pi / 2.0

# Obstacles are padded by the creature's radius in the simulator's collision test, and
# random rectangles are at least this thick.
_AGENT_RADIUS = 5.0
_MIN_OBSTACLE = 30.0

# A step blocked by an obstacle this many ticks in a row means the pulls are balanced
# against a wall, so the agent follows the wall for _ESCAPE_SECONDS instead.
_STUCK_TICKS = 3
_ESCAPE_SECONDS = 1.5

# after an agent sees a predator, it scans at scan_alert_interval for this long
_ALERT_SECONDS = 20.0

# Predator perception, mirrored from src/elements/predator.py and Creature defaults: a 60
# degree cone out to 250 px, and 60 px of hearing all round.
_PRED_HALF_CONE = math.pi / 6.0
_PRED_VISION = 250.0
_PRED_HEARING = 60.0

# How long the higher id wanders after seeing a lower one. Deliberately a flat constant
# rather than a trained parameter.
_CONFLICT_SECONDS = 5.0

# candidate headings when choosing the least-crowded direction to explore
_EXPLORE_DIRS = 16


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


def _face_blocks(e, px: float, py: float, r: float = _AGENT_RADIUS) -> bool:
    """
    Would an agent centred at (px, py) collide with face e = (x1, y1, x2, y2, nx, ny)?
    Mirrors the simulator: obstacles are padded by the radius on every side, so the
    blocked zone behind a face is a slab, square at the corners.
    """
    ux, uy = e[2] - e[0], e[3] - e[1]
    length = math.hypot(ux, uy)
    if length < 1e-6:
        return False
    t = (ux * (px - e[0]) + uy * (py - e[1])) / length
    if t < -r or t > length + r:
        return False
    dn = e[4] * (px - e[0]) + e[5] * (py - e[1])
    return -_MIN_OBSTACLE < dn < r


def _blocking_edge(edges, move_dir: float, dist: float):
    """
    The visible edge an agent-local step of `dist` along `move_dir` collides with, by the
    simulator's own rule, else None. (An 8 px proximity margin used to stand in for this:
    it flagged agents walking alongside a wall, and every step inside an escape gap.)
    """
    ex, ey = dist * math.cos(move_dir), dist * math.sin(move_dir)
    for e in edges:
        (x1, y1), (x2, y2) = e
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        nx, ny = -dy / length, dx / length
        if nx * -x1 + ny * -y1 < 0.0:
            nx, ny = -nx, -ny               # the open side faces the agent
        if (_face_blocks((x1, y1, x2, y2, nx, ny), ex, ey)
                or _segments_cross((0.0, 0.0), (ex, ey), e[0], e[1])):
            return e
    return None


class AgentState:
    """
    The timers one agent carries between ticks. Every bearing is relative to the agent's
    current heading and is rotated by the commanded turn at the end of each tick.
    """

    __slots__ = ("aid", "explore_rel", "explore_until", "dir_until", "food_t",
                 "scan_left", "scan_last", "pred_seen_t", "blocked_ticks",
                 "escape_until", "escape_rel", "last_spawn")

    def __init__(self, aid: int, t: float, rng: random.Random):
        self.aid = aid
        self.explore_rel = rng.uniform(-math.pi, math.pi)  # bearing we are sweeping along
        self.explore_until = -1.0   # gave way to a lower id: explore until this time
        self.dir_until = -1.0       # keep the current exploration bearing until then
        self.food_t = t             # last time a fruit or tree was in view
        self.scan_left = 0.0        # radians of the current look-around still to turn
        self.scan_last = -1e9       # time of the last look-around; -1e9 until first scheduled
        self.pred_seen_t = -1e9     # last time this agent saw a predator
        self.blocked_ticks = 0
        self.escape_until = -1.0
        self.escape_rel = 0.0       # wall-following bearing
        self.last_spawn = -1e9      # time of this agent's last spawn request


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

    The name is historical: this no longer shares anything between agents beyond the
    population count. Each agent is decided independently.
    """

    def __init__(self, params: Optional[Params] = None, rng_seed: int = 0):
        p = params if params is not None else Params()
        # BOUNDS are the contract: a checkpoint trained under older, wider bounds is held
        # to them
        self.p = Params.from_vector(Params.clip_vector(p.to_vector()))
        self._rng_seed = rng_seed
        self.reset()

    # ------------------------------------------------------------------ episode
    def reset(self) -> None:
        self.state: Dict[int, AgentState] = {}
        self.time = -1.0
        self.rng = random.Random(self._rng_seed)

    def _maybe_new_episode(self, sim_time: float) -> None:
        if sim_time < self.time:
            self.reset()
        self.time = sim_time

    # ------------------------------------------------------------------ main
    def decide(self, agent_status: List[dict], sim_time: float) -> List[dict]:
        self._maybe_new_episode(sim_time)
        if not agent_status:
            return []

        alive = set()
        for st in agent_status:
            aid = st["agent_id"]
            alive.add(aid)
            if aid not in self.state:
                self.state[aid] = AgentState(aid, sim_time, self.rng)
        for dead in [a for a in self.state if a not in alive]:
            del self.state[dead]

        pop = len(agent_status)
        return [self._decide_one(st, self.state[st["agent_id"]], pop)
                for st in agent_status]

    # ------------------------------------------------------------------ per agent
    def _decide_one(self, st: dict, s: AgentState, pop: int) -> dict:
        p = self.p
        aid = st["agent_id"]
        energy, max_e = st["energy"], st["max_energy"]
        speed, sprint = st["speed"], st["sprint_speed"]

        # ---- what this agent can see, this tick, and nothing else
        fruits: List[dict] = []
        mates: List[dict] = []
        edges: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        tree = pred = None
        for o in sorted(st["observations"], key=_obs_key):
            kind = o["type"]
            if kind == "Fruit":
                fruits.append(o)
            elif kind == "Tree":
                if tree is None or o["distance"] < tree["distance"]:
                    tree = o
            elif kind == "Predator":
                if pred is None or o["distance"] < pred["distance"]:
                    pred = o
            elif kind == "Agent":
                mates.append(o)
            elif kind == "Edge":
                edges.append(o["coords"])

        if tree is not None or fruits:
            s.food_t = self.time
        if pred is not None:
            s.pred_seen_t = self.time

        # ---- threat. Only a live sighting says which way a predator faces (rel_dir: our
        # bearing in its frame), and one that cannot perceive us is no reason to run.
        pred_dist = pred_ang = None
        aware = False
        if pred is not None:
            pred_dist, pred_ang = pred["distance"], pred["angle"]
            aware = (pred_dist <= _PRED_HEARING + 20.0
                     or (abs(pred["rel_dir"]) < _PRED_HALF_CONE + p.pred_aware_margin
                         and pred_dist < _PRED_VISION + 30.0))
        threatened = aware and pred_dist < p.danger_radius

        starving = energy < max_e * p.starve_frac
        full = energy >= max_e * p.hunger_full

        # ---- fruit: the nearest one in view. Eating is capped at max_energy, so a full
        # agent leaves it for someone hungrier.
        fruit = None
        if fruits:
            if threatened:
                # about to be driven off the patch - take one already within a single
                # step, which is a flee that collects rather than a detour toward danger
                near = [o for o in fruits if o["distance"] <= speed]
                fruit = min(near, key=lambda o: o["distance"]) if near else None
            elif starving or not full:
                fruit = min(fruits, key=lambda o: o["distance"])

        # ---- crowding: a mate with a lower id is in view, so we are the one that leaves.
        # Re-armed only once the timer runs out, so it is 5 s of wandering per sighting
        # rather than a permanently refreshed exile.
        if (not threatened and self.time >= s.explore_until
                and any(o["id"] < aid for o in mates)):
            s.explore_until = self.time + _CONFLICT_SECONDS
            s.explore_rel = self._open_direction(mates)
            s.dir_until = s.explore_until
        forced = self.time < s.explore_until

        # sweeping is triggered by observed scarcity rather than by a clock: it costs
        # energy, and is only worth paying once nothing edible has been in view for a while
        exploring = forced or (
            fruit is None and tree is None and not threatened
            and p.forage_patience > 0.0 and self.time - s.food_t > p.forage_patience)
        if forced and not starving and not threatened:
            fruit = None            # we are leaving, not grazing on the way out
        if exploring and self.time >= s.dir_until:
            s.explore_rel = self._open_direction(mates)
            s.dir_until = self.time + _CONFLICT_SECONDS

        # ---- steering: superpose weighted vectors in the agent-local frame
        vx = vy = 0.0
        hunger = 1.0 - min(1.0, energy / max(1.0, max_e * p.hunger_full))
        target_dist = None

        if fruit is not None:
            w = p.w_fruit * (0.35 + 0.65 * hunger)
            vx += w * math.cos(fruit["angle"])
            vy += w * math.sin(fruit["angle"])
            target_dist = fruit["distance"]

        # hold station near the nearest visible tree - that is where fruit appears
        at_tree = False
        if tree is not None and not exploring:
            d = tree["distance"]
            at_tree = d <= p.camp_radius
            if not at_tree:
                w = p.w_tree * min(1.0, (d - p.camp_radius) / 60.0)
                vx += w * math.cos(tree["angle"])
                vy += w * math.sin(tree["angle"])

        if aware:
            # inverse-square repulsion, away from the threat
            w = p.w_predator * (p.danger_radius / max(pred_dist, 12.0)) ** 2
            vx -= w * math.cos(pred_ang)
            vy -= w * math.sin(pred_ang)

        # separation. While sweeping, push apart harder so the hive covers ground instead
        # of re-searching the same patch together.
        sep_r = p.explore_separation_radius if exploring else p.separation_radius
        crowded = False
        for o in mates:
            d = o["distance"]
            if d < sep_r and d > 1e-6:
                if d < p.separation_radius:
                    crowded = True
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

        # exploration: a slowly drifting persistent bearing keeps agents spreading out
        s.explore_rel = _wrap(s.explore_rel + self.rng.uniform(-0.15, 0.15))
        if exploring:
            vx += p.w_explore_hungry * math.cos(s.explore_rel)
            vy += p.w_explore_hungry * math.sin(s.explore_rel)

        # ---- resolve into a move
        mag = math.hypot(vx, vy)
        # any nonzero sum used to mean a full-speed step; residual pulls are not worth 5/s
        if mag < max(1e-6, p.move_threshold):
            move_dir = 0.0
            want_move = False
        else:
            move_dir = math.atan2(vy, vx)
            want_move = True

        # standing still costs 0.1 energy/s against 5/s for walking - but not on top of a
        # mate, where idling would cancel the separation push
        if (not threatened and fruit is None and at_tree and not exploring and not crowded
                and energy > max_e * p.idle_energy_frac):
            want_move = False

        if not want_move:
            distance = 0.0
        elif threatened and pred_dist < p.sprint_radius:
            distance = sprint
        else:
            distance = speed
            if target_dist is not None and target_dist < distance:
                # don't overshoot a fruit - the eat radius is only ~10 px
                distance = max(target_dist, 0.0)

        # ---- obstacles: a step that keeps running into the same wall means the pulls are
        # balanced against it, and the simulator's sideways slide just jitters in place.
        # Follow the wall for a moment instead.
        blk = None
        if want_move and distance > 1e-6 and edges:
            blk = _blocking_edge(edges, move_dir, distance)
        s.blocked_ticks = s.blocked_ticks + 1 if blk is not None else 0
        if want_move and distance > 1e-6:
            if self.time < s.escape_until:
                move_dir = s.escape_rel
                if edges and _blocking_edge(edges, move_dir, distance) is not None:
                    # dead end along this wall: follow it the other way
                    s.escape_rel = _wrap(s.escape_rel + math.pi)
                    move_dir = s.escape_rel
            elif blk is not None and s.blocked_ticks >= _STUCK_TICKS:
                (x1, y1), (x2, y2) = blk
                along = math.atan2(y2 - y1, x2 - x1)
                if abs(_wrap(along - move_dir)) > HALF_PI:
                    along = _wrap(along + math.pi)
                cx, cy = _closest_point_on_edge(blk)
                away = math.atan2(-cy, -cx)
                move_dir = _wrap(along + 0.25 * _wrap(away - along))
                s.escape_rel = move_dir
                s.escape_until = self.time + _ESCAPE_SECONDS
                s.blocked_ticks = 0

        # ---- look-around: periodically sweep the cone through a full circle. Scans are
        # staggered by agent id so agents never go blind in one direction together.
        if p.scan_interval > 0.0:
            # look around more often while predators are about - they charge from behind
            interval = p.scan_interval
            if self.time - s.pred_seen_t < _ALERT_SECONDS:
                interval = min(interval, p.scan_alert_interval)
            if s.scan_last < -1e8:
                s.scan_last = self.time - interval * ((aid * 0.618034) % 1.0)
            if threatened or fruit is not None:
                s.scan_left = 0.0
            elif s.scan_left <= 0.0 and self.time - s.scan_last >= interval:
                s.scan_left = TWO_PI - st["vision_angle"]
                s.scan_last = self.time

        # ---- heading: independent of travel direction
        if pred_ang is not None:
            # A predator only charges an agent facing more than 90 degrees away from it (or
            # within 90 px); otherwise it pivots around it. Keep it centred in the cone,
            # not just in the forward half-plane: at the edge of a ~30 degree half-cone it
            # drops out of sight.
            tol = 0.25 * st["vision_angle"]
            turn = pred_ang if abs(pred_ang) > tol else 0.0
        elif s.scan_left > 0.0:
            # step one cone width per tick, so consecutive views tile without gaps
            turn = min(s.scan_left, max(0.2, st["vision_angle"]))
            s.scan_left -= turn
        elif want_move:
            # otherwise point roughly where we are walking so the cone scans ahead
            turn = max(-0.45, min(0.45, move_dir * 0.5))
        else:
            turn = 0.0

        # the heading moves by `turn` this tick, so every bearing we hold relative to it
        # moves the other way
        s.explore_rel = _wrap(s.explore_rel - turn)
        s.escape_rel = _wrap(s.escape_rel - turn)

        return {
            "agent_id": aid,
            "move_distance": distance,
            "move_direction": move_dir,
            "turn_angle": turn,
            "spawn_agent": self._want_spawn(st, s, pop, energy),
        }

    # ------------------------------------------------------------------ helpers
    def _open_direction(self, mates: List[dict]) -> float:
        """
        The bearing with the fewest visible mates along it, so explorers spread out
        instead of sweeping the same ground. Candidates start from a random offset, or
        an agent that can see nobody would always pick straight ahead.
        """
        step = TWO_PI / _EXPLORE_DIRS
        base = self.rng.uniform(0.0, step)
        best_a, best_score = base, None
        for k in range(_EXPLORE_DIRS):
            a = _wrap(base + k * step)
            score = 0.0
            for o in mates:
                c = math.cos(_wrap(o["angle"] - a))
                if c > 0.0:                  # only headings pointing toward a mate cost
                    score += c * (300.0 / max(o["distance"], 1.0))
            if best_score is None or score < best_score:
                best_a, best_score = a, score
        return best_a

    def _want_spawn(self, st: dict, s: AgentState, pop: int, energy: float) -> bool:
        """
        Breed below the population target, or whenever the agent is full enough that the
        energy would be thrown away. One child per spawn_cooldown per parent, so a rich
        patch cannot turn into a burst of children that all starve in one place.
        """
        p = self.p
        if self.time - s.last_spawn < p.spawn_cooldown:
            return False
        full = energy >= st["max_energy"] * p.spawn_full_frac
        if not (full or (energy >= p.spawn_energy and pop < p.pop_target)):
            return False
        if pop >= p.pop_max:
            return False
        s.last_spawn = self.time
        return True
