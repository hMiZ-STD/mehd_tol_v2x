"""
ev_preemption.py
EV speed boost and path clearing only.
TLS control is fully handled by corridor_ev_manager.py
"""

import traci
from config import EV_TYPE_ID, EV_MAX_SPEED

ev_travel_log: dict = {}


def _force_yield(ev_id):
    try:
        ev_route = traci.vehicle.getRoute(ev_id)
        ev_idx   = traci.vehicle.getRouteIndex(ev_id)
        ev_edge  = traci.vehicle.getRoadID(ev_id)
        ev_pos   = traci.vehicle.getLanePosition(ev_id)
        if ev_edge.startswith(":"):
            return
        upcoming = set(ev_route[ev_idx: ev_idx + 5])
        for other_id in traci.vehicle.getIDList():
            if other_id == ev_id:
                continue
            try:
                if traci.vehicle.getTypeID(other_id) == EV_TYPE_ID:
                    continue
                other_edge = traci.vehicle.getRoadID(other_id)
                if other_edge not in upcoming:
                    continue
                if other_edge == ev_edge:
                    if traci.vehicle.getLanePosition(other_id) <= ev_pos:
                        continue
                max_spd = traci.vehicle.getMaxSpeed(other_id)
                traci.vehicle.setSpeed(other_id, max_spd * 0.8)
            except traci.TraCIException:
                pass
    except traci.TraCIException:
        pass


def apply_ev_preemption():
    all_vehicles = traci.vehicle.getIDList()
    active_evs   = []

    for veh_id in all_vehicles:
        try:
            if traci.vehicle.getTypeID(veh_id) != EV_TYPE_ID:
                continue
        except traci.TraCIException:
            continue

        active_evs.append(veh_id)

        if veh_id not in ev_travel_log:
            ev_travel_log[veh_id] = {
                "entry_time" : traci.simulation.getTime(),
                "exit_time"  : None,
                "travel_time": None,
            }

        try:
            traci.vehicle.setSpeed(veh_id, EV_MAX_SPEED)
            _force_yield(veh_id)
        except traci.TraCIException:
            pass

    for veh_id in list(ev_travel_log.keys()):
        info = ev_travel_log[veh_id]
        if veh_id not in all_vehicles and info["exit_time"] is None:
            info["exit_time"]   = traci.simulation.getTime()
            info["travel_time"] = info["exit_time"] - info["entry_time"]

    return active_evs


def get_ev_summary():
    completed = [v["travel_time"] for v in ev_travel_log.values() if v["travel_time"] is not None]
    return round(sum(completed) / len(completed), 2) if completed else 0.0
