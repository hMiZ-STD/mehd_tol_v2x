"""
rerouter.py — Continuous Vehicle Rerouting

EVs are skipped (they follow fixed emergency routes).
Distance-weighted candidate selection spreads traffic across network.
"""

import traci
import random
import math
from config       import EV_TYPE_ID
from network_graph import get_trusted_edges, get_trusted_set

_DISTANT_SAMPLE = 40
_DISTANT_TRY    = 20
_FALLBACK_TRY   = 10


def _edge_midpoint(edge_id: str):
    try:
        n = traci.edge.getLaneNumber(edge_id)
        if n == 0:
            return None
        shape = traci.lane.getShape(f"{edge_id}_0")
        if not shape:
            return None
        xs = [p[0] for p in shape]
        ys = [p[1] for p in shape]
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    except traci.TraCIException:
        return None


def _try_reroute(veh_id: str, candidates: list) -> bool:
    for dest in candidates:
        try:
            traci.vehicle.changeTarget(veh_id, dest)
            traci.vehicle.rerouteTraveltime(veh_id)
            return True
        except traci.TraCIException:
            continue
    return False


def apply_rerouting() -> None:
    trusted     = get_trusted_edges()
    trusted_set = get_trusted_set()
    if not trusted:
        return

    for veh_id in traci.vehicle.getIDList():
        try:
            # Skip emergency vehicles — they follow fixed preemption routes
            if traci.vehicle.getTypeID(veh_id) == EV_TYPE_ID:
                continue

            route        = traci.vehicle.getRoute(veh_id)
            idx          = traci.vehicle.getRouteIndex(veh_id)
            current_edge = traci.vehicle.getRoadID(veh_id)

            if len(route) - idx > 3:
                continue
            if current_edge.startswith(":"):
                continue
            if current_edge not in trusted_set:
                continue

            candidates = [e for e in trusted if e != current_edge]
            if not candidates:
                continue

            cur_pos = _edge_midpoint(current_edge)

            if cur_pos and len(candidates) > _DISTANT_SAMPLE:
                sample = random.sample(candidates, _DISTANT_SAMPLE)
                def dist_key(e):
                    p = _edge_midpoint(e)
                    return math.hypot(p[0] - cur_pos[0], p[1] - cur_pos[1]) if p else 0.0
                sample.sort(key=dist_key, reverse=True)
                primary  = sample[:_DISTANT_TRY]
                fallback = sample[_DISTANT_TRY:]
            else:
                random.shuffle(candidates)
                primary  = candidates[:_DISTANT_TRY]
                fallback = candidates[_DISTANT_TRY:_DISTANT_TRY + _FALLBACK_TRY]

            if not _try_reroute(veh_id, primary):
                _try_reroute(veh_id, fallback)

        except traci.TraCIException:
            pass
