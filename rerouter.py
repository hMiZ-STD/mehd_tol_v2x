"""
Continuous vehicle rerouting with safety guards.

Design goals:
- Never reroute emergency vehicles (EVs follow fixed preemption routes).
- Avoid TraCI changeTarget unreachable spam by pre-checking route reachability.
- Throttle repeated reroute attempts per vehicle with a cooldown.
- Keep candidate selection distance-aware to spread traffic.
"""

from __future__ import annotations

import math
import random
from typing import Dict, Optional, Tuple

import traci

from config import EV_TYPE_ID
from network_graph import get_trusted_edges, get_trusted_set

# Candidate selection tuning.
_DISTANT_SAMPLE = 40
_DISTANT_TRY = 20
_FALLBACK_TRY = 10

# Runtime throttling tuning.
_COOLDOWN_STEPS = 20
_FAIL_WINDOW_STEPS = 200
_MAX_FAILURES_IN_WINDOW = 4

# In-memory state keyed by vehicle id.
_last_attempt_step: Dict[str, int] = {}
_failure_count: Dict[str, int] = {}
_failure_window_start: Dict[str, int] = {}


def _edge_midpoint(edge_id: str) -> Optional[Tuple[float, float]]:
    """Return lane-0 midpoint for distance scoring, or None if unavailable."""
    try:
        lane_count = traci.edge.getLaneNumber(edge_id)
        if lane_count <= 0:
            return None
        shape = traci.lane.getShape(f"{edge_id}_0")
        if not shape:
            return None
        xs = [p[0] for p in shape]
        ys = [p[1] for p in shape]
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    except traci.TraCIException:
        return None


def _is_reachable(from_edge: str, to_edge: str) -> bool:
    """Check whether SUMO can produce a route from current edge to target edge."""
    if from_edge == to_edge:
        return False
    if from_edge.startswith(":") or to_edge.startswith(":"):
        return False
    try:
        route = traci.simulation.findRoute(from_edge, to_edge)
        return bool(route and route.edges)
    except traci.TraCIException:
        return False


def _can_attempt_reroute(veh_id: str, sim_step: int) -> bool:
    """
    Gate reroute attempts per vehicle to reduce noisy repeated failures.

    - Cooldown between attempts.
    - Failure-window ceiling to avoid thrashing on problematic vehicles.
    """
    last = _last_attempt_step.get(veh_id)
    if last is not None and sim_step - last < _COOLDOWN_STEPS:
        return False

    window_start = _failure_window_start.get(veh_id, sim_step)
    fail_count = _failure_count.get(veh_id, 0)

    if sim_step - window_start > _FAIL_WINDOW_STEPS:
        _failure_window_start[veh_id] = sim_step
        _failure_count[veh_id] = 0
        return True

    if fail_count >= _MAX_FAILURES_IN_WINDOW:
        return False
    return True


def _mark_attempt(veh_id: str, sim_step: int, success: bool) -> None:
    _last_attempt_step[veh_id] = sim_step
    if success:
        _failure_count[veh_id] = 0
        _failure_window_start[veh_id] = sim_step
        return

    if veh_id not in _failure_window_start:
        _failure_window_start[veh_id] = sim_step
    _failure_count[veh_id] = _failure_count.get(veh_id, 0) + 1


def _try_reroute(veh_id: str, current_edge: str, candidates: list[str]) -> bool:
    """Attempt reroute across candidates with reachability pre-check."""
    for dest in candidates:
        if not _is_reachable(current_edge, dest):
            continue
        try:
            traci.vehicle.changeTarget(veh_id, dest)
            traci.vehicle.rerouteTraveltime(veh_id)
            return True
        except traci.TraCIException:
            # Destination became invalid or vehicle state changed mid-step.
            continue
    return False


def _build_candidate_buckets(current_edge: str, trusted: list[str]) -> tuple[list[str], list[str]]:
    """
    Build primary/fallback candidate buckets.

    Primary bucket prioritizes farther edges from current position to spread load.
    """
    candidates = [e for e in trusted if e != current_edge and not e.startswith(":")]
    if not candidates:
        return [], []

    cur_pos = _edge_midpoint(current_edge)
    if cur_pos and len(candidates) > _DISTANT_SAMPLE:
        sample = random.sample(candidates, _DISTANT_SAMPLE)

        def dist_key(edge_id: str) -> float:
            p = _edge_midpoint(edge_id)
            if not p:
                return 0.0
            return math.hypot(p[0] - cur_pos[0], p[1] - cur_pos[1])

        sample.sort(key=dist_key, reverse=True)
        primary = sample[:_DISTANT_TRY]
        fallback = sample[_DISTANT_TRY : _DISTANT_TRY + _FALLBACK_TRY]
        return primary, fallback

    random.shuffle(candidates)
    primary = candidates[:_DISTANT_TRY]
    fallback = candidates[_DISTANT_TRY : _DISTANT_TRY + _FALLBACK_TRY]
    return primary, fallback


def apply_rerouting() -> None:
    """Reroute near-destination non-EV vehicles onto trusted edges."""
    trusted = get_trusted_edges()
    trusted_set = get_trusted_set()
    if not trusted:
        return

    try:
        sim_step = int(traci.simulation.getTime())
    except traci.TraCIException:
        sim_step = 0

    for veh_id in traci.vehicle.getIDList():
        try:
            # Skip emergency vehicles.
            if traci.vehicle.getTypeID(veh_id) == EV_TYPE_ID:
                continue

            if not _can_attempt_reroute(veh_id, sim_step):
                continue

            route = traci.vehicle.getRoute(veh_id)
            idx = traci.vehicle.getRouteIndex(veh_id)
            current_edge = traci.vehicle.getRoadID(veh_id)

            # Only reroute vehicles close to route end to avoid unnecessary churn.
            if len(route) - idx > 3:
                continue
            if current_edge.startswith(":"):
                continue
            if current_edge not in trusted_set:
                continue

            primary, fallback = _build_candidate_buckets(current_edge, trusted)
            if not primary and not fallback:
                continue

            success = _try_reroute(veh_id, current_edge, primary)
            if not success and fallback:
                success = _try_reroute(veh_id, current_edge, fallback)

            _mark_attempt(veh_id, sim_step, success)
        except traci.TraCIException:
            _mark_attempt(veh_id, sim_step, False)
            continue

