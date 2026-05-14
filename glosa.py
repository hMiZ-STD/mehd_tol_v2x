"""
glosa.py — Green Light Optimal Speed Advisory (GLOSA)

All fixes applied:
Level 1 — Natural arrival check: don't slow vehicles that arrive on green anyway
Level 1 — Min speed 5 m/s: prevents sub-18 km/h crawlers
Level 2 — Lead vehicle only: followers handled by SUMO car-following model
Level 3 — Green checked FIRST: instant release on signal flip, zero lag
         — time_to_switch <= 0 guard: catches exact step signal turns green
"""

import traci
from config import GLOSA_LOOKAHEAD, GLOSA_MIN_SPEED, GLOSA_MAX_SPEED

_STOP_STATES = frozenset("rRyYsS")
_GO_STATES   = frozenset("gGuU")


def _is_lead_vehicle(veh_id: str, my_lane: str, my_pos: float) -> bool:
    """
    Returns True only if no other vehicle is AHEAD of this one on the same lane.
    Only the lead vehicle needs a GLOSA advisory — all followers slow down
    naturally via SUMO's built-in car-following model (Krauss/IDM).
    This prevents accordion waves from multiple simultaneous advisories.
    """
    try:
        for other_id in traci.vehicle.getIDList():
            if other_id == veh_id:
                continue
            try:
                if traci.vehicle.getLaneID(other_id) != my_lane:
                    continue
                if traci.vehicle.getLanePosition(other_id) > my_pos:
                    return False  # Someone ahead on same lane — skip
            except traci.TraCIException:
                continue
        return True
    except traci.TraCIException:
        return False


def apply_glosa(veh_id: str) -> None:
    """
    Physics-correct GLOSA advisory for a single non-EV vehicle.

    Decision flow:
    1. Skip if already stopped
    2. Skip if no TLS ahead within GLOSA_LOOKAHEAD
    3. Skip if not the lead vehicle on this lane (Level 2)
    4. GREEN → release immediately (Level 3 — checked FIRST)
    5. time_to_switch <= 0 → release (catches exact flip step)
    6. RED/YELLOW:
       a. Natural arrival after green? → release (Level 1)
       b. Signal switching in ≤ 2s?   → release (roll through)
       c. Too close to physically stop? → release (SUMO handles)
       d. Otherwise → advisory = dist / (time + 0.5s buffer)
          clamped to [GLOSA_MIN_SPEED, GLOSA_MAX_SPEED]
    """
    try:
        speed = traci.vehicle.getSpeed(veh_id)
        if speed < 0.5:
            return  # Already stopped — no advisory needed

        tls_info = traci.vehicle.getNextTLS(veh_id)
        if not tls_info:
            traci.vehicle.setSpeed(veh_id, -1)
            return

        tls_id, _, dist_to_tls, tls_state = tls_info[0]

        # Outside advisory range — release and return
        if dist_to_tls > GLOSA_LOOKAHEAD or dist_to_tls < 1.0:
            traci.vehicle.setSpeed(veh_id, -1)
            return

        # Level 2: only advise the lead vehicle on this lane
        my_lane = traci.vehicle.getLaneID(veh_id)
        my_pos  = traci.vehicle.getLanePosition(veh_id)
        if not _is_lead_vehicle(veh_id, my_lane, my_pos):
            traci.vehicle.setSpeed(veh_id, -1)
            return

        # Get time remaining in current phase
        time_to_switch = (
            traci.trafficlight.getNextSwitch(tls_id)
            - traci.simulation.getTime()
        )

        # ── Level 3: GREEN CHECK FIRST — always release immediately ──────────
        if tls_state in _GO_STATES:
            traci.vehicle.setSpeed(veh_id, -1)
            return

        # ── Level 3: signal just flipped (time_to_switch <= 0) ───────────────
        # Catches the exact simulation step the signal changes state.
        # Without this, the advisory from the previous step persists for 1 step.
        if time_to_switch <= 0:
            traci.vehicle.setSpeed(veh_id, -1)
            return

        # ── RED / YELLOW / STOP handling ──────────────────────────────────────
        if tls_state in _STOP_STATES:

            # Level 1: will vehicle naturally arrive AFTER signal turns green?
            # If yes — it will cross on green without slowing. Don't touch it.
            natural_arrival = dist_to_tls / max(speed, 0.1)
            if natural_arrival >= time_to_switch:
                traci.vehicle.setSpeed(veh_id, -1)
                return

            # Signal switching very soon — vehicle rolls through on green
            if 0 < time_to_switch <= 2.0:
                traci.vehicle.setSpeed(veh_id, -1)
                return

            # Physics: minimum distance needed to stop from current speed
            # Formula: d_min = v² / (2 × a)
            decel      = max(traci.vehicle.getDecel(veh_id), 0.5)
            min_stop_d = (speed ** 2) / (2.0 * decel)

            # Too close to stop safely (85% margin) — hand off to SUMO
            if dist_to_tls <= min_stop_d * 0.85:
                traci.vehicle.setSpeed(veh_id, -1)
                return

            # All checks passed — safe to issue advisory
            # + 0.5 s reaction/communication latency buffer
            advisory = dist_to_tls / max(time_to_switch + 0.5, 1.0)
            advisory = max(GLOSA_MIN_SPEED, min(advisory, GLOSA_MAX_SPEED))
            traci.vehicle.setSpeed(veh_id, advisory)

        else:
            # Unknown / blinking / off state — always release
            traci.vehicle.setSpeed(veh_id, -1)

    except traci.TraCIException:
        pass
