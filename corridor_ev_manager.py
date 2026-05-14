"""
corridor_ev_manager.py
Full-route green wave for EVs.
- On EV spawn  : ALL signals on entire route go green immediately + route drawn RED
- Each step    : as EV passes each junction, restore signal + remove red line
- On EV depart : restore all remaining signals + remove all red lines
"""

import traci
from config import EV_TYPE_ID

_ev_corridor_state: dict = {}
_protected_tls:     set  = set()
_NON_GREEN = frozenset("rRyYsSuUoO")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_green(state: str) -> str:
    return "".join("G" if c in _NON_GREEN else c for c in state)


def _add_route_visual(ev_id: str, route_edges: list):
    """Draw thick red lines on every edge of EV route in SUMO GUI."""
    for edge in route_edges:
        poly_id = f"ev_route_{ev_id}_{edge}"
        try:
            shape = traci.lane.getShape(f"{edge}_0")
            if not shape:
                continue
            traci.polygon.add(
                poly_id,
                shape,
                color=(255, 0, 0, 220),
                fill=False,
                layer=100,
                lineWidth=5,
            )
        except traci.TraCIException:
            pass


def _remove_edge_visual(ev_id: str, edge: str):
    """Remove red line from one edge after EV passes it."""
    poly_id = f"ev_route_{ev_id}_{edge}"
    try:
        traci.polygon.remove(poly_id)
    except traci.TraCIException:
        pass


def _remove_all_route_visual(ev_id: str, route_edges: list):
    """Remove all red lines when EV departs."""
    for edge in route_edges:
        _remove_edge_visual(ev_id, edge)


def _get_tls_on_route(ev_id: str) -> list:
    """
    Returns ordered list of (tls_id, controlling_edge)
    for every signal on the EV's full remaining route.
    """
    result = []
    try:
        route     = traci.vehicle.getRoute(ev_id)
        idx       = traci.vehicle.getRouteIndex(ev_id)
        remaining = route[idx:]
    except traci.TraCIException:
        return result

    seen = set()
    for edge in remaining:
        try:
            tls_list = traci.trafficlight.getIDList()
        except traci.TraCIException:
            break
        for tls_id in tls_list:
            if tls_id in seen:
                continue
            try:
                controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
                for lane in controlled_lanes:
                    if lane and lane.rsplit("_", 1)[0] == edge:
                        seen.add(tls_id)
                        result.append((tls_id, edge))
                        break
            except traci.TraCIException:
                continue
    return result


def _ev_passed_tls(ev_id: str, tls_edge: str) -> bool:
    """Returns True if EV has already moved past this edge."""
    try:
        route        = traci.vehicle.getRoute(ev_id)
        idx          = traci.vehicle.getRouteIndex(ev_id)
        current_edge = traci.vehicle.getRoadID(ev_id)
    except traci.TraCIException:
        return False

    if current_edge.startswith(":"):
        return False

    try:
        tls_idx = route.index(tls_edge)
        cur_idx = route.index(current_edge) if current_edge in route else idx
        return cur_idx > tls_idx
    except ValueError:
        return False


# ── Core corridor actions ─────────────────────────────────────────────────────

def _green_entire_corridor(ev_id: str):
    """Set ALL signals on EV route to green + draw red route lines."""
    info = _ev_corridor_state.get(ev_id)
    if not info:
        return

    for tls_id, _ in info["route_tls"]:
        if tls_id in info["greened"]:
            continue
        try:
            current = traci.trafficlight.getRedYellowGreenState(tls_id)
            if tls_id not in info["original"]:
                info["original"][tls_id] = current
            traci.trafficlight.setRedYellowGreenState(tls_id, _safe_green(current))
            traci.trafficlight.setPhaseDuration(tls_id, 999.0)
            info["greened"].add(tls_id)
        except traci.TraCIException:
            pass

    # Draw red route visualization
    all_edges = [edge for _, edge in info["route_tls"]]
    _add_route_visual(ev_id, all_edges)

    tls_count = len(info["greened"])
    if tls_count > 0:
        print(f"  🚑 [EV CORRIDOR] {ev_id} → {tls_count} signals cleared, route marked RED")


def _restore_passed_junctions(ev_id: str):
    """Restore signals + remove red lines for edges EV has already passed."""
    info = _ev_corridor_state.get(ev_id)
    if not info:
        return

    for tls_id, tls_edge in info["route_tls"]:
        if tls_id in info["restored"]:
            continue
        if not _ev_passed_tls(ev_id, tls_edge):
            continue
        try:
            traci.trafficlight.setProgram(tls_id, "0")
            info["restored"].add(tls_id)
            _remove_edge_visual(ev_id, tls_edge)   # remove red line from passed edge
            print(f"  ✅ [EV CORRIDOR] Restored: {tls_id} (EV passed)")
        except traci.TraCIException:
            try:
                orig = info["original"].get(tls_id)
                if orig:
                    traci.trafficlight.setRedYellowGreenState(tls_id, orig)
                info["restored"].add(tls_id)
                _remove_edge_visual(ev_id, tls_edge)
            except traci.TraCIException:
                pass


def _restore_all_for_ev(ev_id: str):
    """EV departed — restore all remaining signals + remove all red lines."""
    info = _ev_corridor_state.get(ev_id)
    if not info:
        return

    for tls_id, _ in info["route_tls"]:
        if tls_id in info["restored"]:
            continue
        try:
            traci.trafficlight.setProgram(tls_id, "0")
        except traci.TraCIException:
            try:
                orig = info["original"].get(tls_id)
                if orig:
                    traci.trafficlight.setRedYellowGreenState(tls_id, orig)
            except traci.TraCIException:
                pass
        info["restored"].add(tls_id)

    # Remove all remaining red route lines
    all_edges = [edge for _, edge in info["route_tls"]]
    _remove_all_route_visual(ev_id, all_edges)

    print(f"  🏁 [EV CORRIDOR] {ev_id} cleared corridor — signals restored, route unmarked")
    del _ev_corridor_state[ev_id]


# ── Main step function (called from main.py) ──────────────────────────────────

def update_ev_corridors() -> set:
    """
    Call every simulation step.
    Returns set of currently protected TLS IDs
    so adaptive_signals.py skips them.
    """
    global _protected_tls

    active_evs = set()
    for veh_id in traci.vehicle.getIDList():
        try:
            if traci.vehicle.getTypeID(veh_id) != EV_TYPE_ID:
                continue
        except traci.TraCIException:
            continue
        active_evs.add(veh_id)

        if veh_id not in _ev_corridor_state:
            route_tls = _get_tls_on_route(veh_id)
            _ev_corridor_state[veh_id] = {
                "route_tls": route_tls,
                "greened"  : set(),
                "restored" : set(),
                "original" : {},
            }
            _green_entire_corridor(veh_id)
        else:
            _restore_passed_junctions(veh_id)

    # Handle EVs that departed this step
    departed = set(_ev_corridor_state.keys()) - active_evs
    for ev_id in departed:
        _restore_all_for_ev(ev_id)

    # Protected = greened but not yet restored
    protected = set()
    for info in _ev_corridor_state.values():
        protected.update(info["greened"] - info["restored"])

    _protected_tls = protected
    return set(_protected_tls)


def is_tls_protected(tls_id: str) -> bool:
    return tls_id in _protected_tls


def get_protected_tls() -> set:
    return set(_protected_tls)
