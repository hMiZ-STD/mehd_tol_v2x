"""Hybrid GLOSA controller with PPO option and rule-based fallback."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import traci

import config
from config import GLOSA_LOOKAHEAD, GLOSA_MAX_SPEED, GLOSA_MIN_SPEED

TIME_TO_SWITCH_NORM = 60.0
QUEUE_AHEAD_NORM = 10.0

_ppo_model: Any = None
_rl_ready: bool = False

_STOP_STATES = frozenset("rRyYsS")
_GO_STATES = frozenset("gGuU")


def initialize_rl_glosa() -> bool:
    """Load PPO-GLOSA model if available."""
    global _ppo_model, _rl_ready
    model_path = os.path.join(config.MODEL_DIR, "ppo_glosa_final")
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("⚠ PPO-GLOSA model not found — falling back to rule-based GLOSA")
        _rl_ready = False
        return False

    try:
        _ppo_model = PPO.load(model_path)
        _rl_ready = True
        print("✓ PPO-GLOSA controller loaded successfully")
    except Exception:
        print("⚠ PPO-GLOSA model not found — falling back to rule-based GLOSA")
        _rl_ready = False
    return _rl_ready


def _next_tl_info(veh_id: str) -> dict[str, float] | None:
    """Return next traffic-light context for one vehicle."""
    try:
        tls_info = traci.vehicle.getNextTLS(veh_id)
        if not tls_info:
            return None
        tls_id, lane_idx, dist, state = tls_info[0]
        if dist > GLOSA_LOOKAHEAD or dist < 0.0:
            return None

        time_to_switch = (
            traci.trafficlight.getNextSwitch(tls_id) - traci.simulation.getTime()
        )
        lane_id = traci.vehicle.getLaneID(veh_id)
        queue_ahead = float(traci.lane.getLastStepHaltingNumber(lane_id))
        is_green = 1.0 if state in _GO_STATES else 0.0
        del lane_idx
        return {
            "dist": float(dist),
            "time_to_switch": float(max(0.0, time_to_switch)),
            "is_green": float(is_green),
            "queue_ahead": float(queue_ahead),
        }
    except traci.TraCIException:
        return None


def _is_lead_vehicle(veh_id: str, my_lane: str, my_pos: float) -> bool:
    """
    Returns True only if no other vehicle is AHEAD of this one on the same lane.
    Only the lead vehicle needs a GLOSA advisory â€” all followers slow down
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
                    return False  # Someone ahead on same lane â€” skip
            except traci.TraCIException:
                continue
        return True
    except traci.TraCIException:
        return False


def _apply_rule_based_glosa(veh_id: str) -> None:
    """
    Physics-correct GLOSA advisory for a single non-EV vehicle.

    Decision flow:
    1. Skip if already stopped
    2. Skip if no TLS ahead within GLOSA_LOOKAHEAD
    3. Skip if not the lead vehicle on this lane (Level 2)
    4. GREEN â†’ release immediately (Level 3 â€” checked FIRST)
    5. time_to_switch <= 0 â†’ release (catches exact flip step)
    6. RED/YELLOW:
       a. Natural arrival after green? â†’ release (Level 1)
       b. Signal switching in â‰¤ 2s?   â†’ release (roll through)
       c. Too close to physically stop? â†’ release (SUMO handles)
       d. Otherwise â†’ advisory = dist / (time + 0.5s buffer)
          clamped to [GLOSA_MIN_SPEED, GLOSA_MAX_SPEED]
    """
    try:
        speed = traci.vehicle.getSpeed(veh_id)
        if speed < 0.5:
            return  # Already stopped â€” no advisory needed

        tls_info = traci.vehicle.getNextTLS(veh_id)
        if not tls_info:
            traci.vehicle.setSpeed(veh_id, -1)
            return

        tls_id, _, dist_to_tls, tls_state = tls_info[0]

        # Outside advisory range â€” release and return
        if dist_to_tls > GLOSA_LOOKAHEAD or dist_to_tls < 1.0:
            traci.vehicle.setSpeed(veh_id, -1)
            return

        # Level 2: only advise the lead vehicle on this lane
        my_lane = traci.vehicle.getLaneID(veh_id)
        my_pos = traci.vehicle.getLanePosition(veh_id)
        if not _is_lead_vehicle(veh_id, my_lane, my_pos):
            traci.vehicle.setSpeed(veh_id, -1)
            return

        # Get time remaining in current phase
        time_to_switch = (
            traci.trafficlight.getNextSwitch(tls_id) - traci.simulation.getTime()
        )

        # â”€â”€ Level 3: GREEN CHECK FIRST â€” always release immediately â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if tls_state in _GO_STATES:
            traci.vehicle.setSpeed(veh_id, -1)
            return

        # â”€â”€ Level 3: signal just flipped (time_to_switch <= 0) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Catches the exact simulation step the signal changes state.
        # Without this, the advisory from the previous step persists for 1 step.
        if time_to_switch <= 0:
            traci.vehicle.setSpeed(veh_id, -1)
            return

        # â”€â”€ RED / YELLOW / STOP handling â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if tls_state in _STOP_STATES:

            # Level 1: will vehicle naturally arrive AFTER signal turns green?
            # If yes â€” it will cross on green without slowing. Don't touch it.
            natural_arrival = dist_to_tls / max(speed, 0.1)
            if natural_arrival >= time_to_switch:
                traci.vehicle.setSpeed(veh_id, -1)
                return

            # Signal switching very soon â€” vehicle rolls through on green
            if 0 < time_to_switch <= 2.0:
                traci.vehicle.setSpeed(veh_id, -1)
                return

            # Physics: minimum distance needed to stop from current speed
            # Formula: d_min = vÂ² / (2 Ã— a)
            decel = max(traci.vehicle.getDecel(veh_id), 0.5)
            min_stop_d = (speed**2) / (2.0 * decel)

            # Too close to stop safely (85% margin) â€” hand off to SUMO
            if dist_to_tls <= min_stop_d * 0.85:
                traci.vehicle.setSpeed(veh_id, -1)
                return

            # All checks passed â€” safe to issue advisory
            # + 0.5 s reaction/communication latency buffer
            advisory = dist_to_tls / max(time_to_switch + 0.5, 1.0)
            advisory = max(GLOSA_MIN_SPEED, min(advisory, GLOSA_MAX_SPEED))
            traci.vehicle.setSpeed(veh_id, advisory)

        else:
            # Unknown / blinking / off state â€” always release
            traci.vehicle.setSpeed(veh_id, -1)

    except traci.TraCIException:
        pass


def apply_glosa(veh_id: str) -> None:
    """Apply PPO-based or rule-based GLOSA control for a single vehicle."""
    if config.USE_RL_GLOSA and _rl_ready:
        try:
            tl = _next_tl_info(veh_id)
            if tl is None:
                traci.vehicle.setSpeed(veh_id, -1)
                return

            speed = float(traci.vehicle.getSpeed(veh_id))
            obs = np.array(
                [
                    float(np.clip(tl["dist"] / GLOSA_LOOKAHEAD, 0.0, 1.0)),
                    float(np.clip(speed / GLOSA_MAX_SPEED, 0.0, 1.0)),
                    float(np.clip(tl["time_to_switch"] / TIME_TO_SWITCH_NORM, 0.0, 1.0)),
                    float(tl["is_green"]),
                    float(np.clip(tl["queue_ahead"] / QUEUE_AHEAD_NORM, 0.0, 1.0)),
                ],
                dtype=np.float32,
            )
            action, _ = _ppo_model.predict(obs.reshape(1, -1), deterministic=True)
            advisory = float(np.asarray(action).reshape(-1)[0]) * config.GLOSA_MAX_SPEED
            advisory = max(config.GLOSA_MIN_SPEED, min(config.GLOSA_MAX_SPEED, advisory))

            if tl["is_green"] == 1.0 and tl["dist"] > 30.0:
                traci.vehicle.setSpeed(veh_id, -1)
                return

            traci.vehicle.setSpeed(veh_id, advisory)
            return
        except traci.TraCIException:
            pass

    _apply_rule_based_glosa(veh_id)


def get_glosa_mode() -> str:
    """Return active GLOSA mode label."""
    return "PPO-GLOSA" if (_rl_ready and config.USE_RL_GLOSA) else "Rule-Based-GLOSA"
