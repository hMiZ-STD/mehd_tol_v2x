"""Hybrid adaptive signal controller with RL-DQN + rule-based fallback."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import traci

import config
from config import (
    ADAPTIVE_SIGNALS_ENABLED,
    AS_ACTION_COOLDOWN,
    AS_EMPTY_QUEUE,
    AS_EXTEND_QUEUE,
    AS_EXTEND_STEP,
    AS_MAX_GREEN,
    AS_MIN_GREEN,
    AS_SWITCH_QUEUE,
)

QUEUE_NORM_DENOMINATOR = 20.0
SPEED_NORM_DENOMINATOR = 13.89
PHASE_TIME_NORM_DENOMINATOR = 30.0
NS_ANGLE_RANGES = ((315.0, 360.0), (0.0, 45.0), (135.0, 225.0))
TRAINED_OBS_SIZE = 24

_dqn_model: Any = None
_phase_timers: dict[str, int] = {}
_rl_ready: bool = False
_sorted_tl_ids: list[str] = []

_phase_tracker: dict = {}
_last_action_time: dict = {}


def initialize_rl_controller() -> bool:
    """Load DQN model for signal control if available."""
    global _dqn_model, _rl_ready
    model_path = os.path.join(config.MODEL_DIR, "dqn_signal_final")
    try:
        from stable_baselines3 import DQN
    except ImportError:
        print(
            f"⚠ DQN model not found at {model_path} — falling back to rule-based control"
        )
        _rl_ready = False
        return False

    try:
        _dqn_model = DQN.load(model_path)
        _rl_ready = True
        print("✓ DQN signal controller loaded successfully")
    except Exception:
        print(
            f"⚠ DQN model not found at {model_path} — falling back to rule-based control"
        )
        _rl_ready = False
    return _rl_ready


def _is_ns_edge(edge_id: str) -> bool:
    try:
        angle = float(traci.edge.getAngle(edge_id)) % 360.0
    except traci.TraCIException:
        return False

    for low, high in NS_ANGLE_RANGES:
        if low <= angle <= high:
            return True
    return False


def _build_obs(tl_ids: list[str]) -> np.ndarray:
    """Build RL observation with identical feature logic as training environment."""
    features: list[float] = []
    ordered_tl_ids = sorted(tl_ids)  # CRITICAL: must match training order
    for tl_id in ordered_tl_ids:
        ns_lanes: set[str] = set()
        ew_lanes: set[str] = set()
        try:
            controlled_links = traci.trafficlight.getControlledLinks(tl_id)
        except traci.TraCIException:
            controlled_links = []

        for signal_links in controlled_links:
            for link in signal_links:
                if not link:
                    continue
                incoming_lane = link[0]
                if not incoming_lane or incoming_lane.startswith(":"):
                    continue
                edge_id = incoming_lane.rsplit("_", 1)[0]
                if _is_ns_edge(edge_id):
                    ns_lanes.add(incoming_lane)
                else:
                    ew_lanes.add(incoming_lane)

        ns_list = sorted(ns_lanes)
        ew_list = sorted(ew_lanes)

        queue_ns = (
            sum(traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in ns_list)
            / QUEUE_NORM_DENOMINATOR
            if ns_list
            else 0.0
        )
        queue_ew = (
            sum(traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in ew_list)
            / QUEUE_NORM_DENOMINATOR
            if ew_list
            else 0.0
        )

        ns_speeds = []
        for lane_id in ns_list:
            vehicle_count = traci.lane.getLastStepVehicleNumber(lane_id)
            if vehicle_count == 0:
                ns_speeds.append(0.0)
            else:
                ns_speeds.append(
                    traci.lane.getLastStepMeanSpeed(lane_id) / SPEED_NORM_DENOMINATOR
                )
        speed_ns = float(np.mean(ns_speeds)) if ns_speeds else 0.0

        ew_speeds = []
        for lane_id in ew_list:
            vehicle_count = traci.lane.getLastStepVehicleNumber(lane_id)
            if vehicle_count == 0:
                ew_speeds.append(0.0)
            else:
                ew_speeds.append(
                    traci.lane.getLastStepMeanSpeed(lane_id) / SPEED_NORM_DENOMINATOR
                )
        speed_ew = float(np.mean(ew_speeds)) if ew_speeds else 0.0

        try:
            current_phase = traci.trafficlight.getPhase(tl_id)
            total_phases = len(traci.trafficlight.getAllProgramLogics(tl_id)[0].phases)
            phase_encoded = (
                current_phase / total_phases if total_phases > 0 else 0.0
            )
        except traci.TraCIException:
            phase_encoded = 0.0

        phase_time_norm = min(
            _phase_timers.get(tl_id, 0) / PHASE_TIME_NORM_DENOMINATOR, 1.0
        )
        features.extend(
            [
                float(queue_ns),
                float(queue_ew),
                float(speed_ns),
                float(speed_ew),
                float(phase_encoded),
                float(phase_time_norm),
            ]
        )

    return np.array(features, dtype=np.float32)


def _phase_count(tls_id: str) -> int:
    try:
        logics = traci.trafficlight.getAllProgramLogics(tls_id)
        if not logics:
            return 0
        return len(logics[0].phases)
    except traci.TraCIException:
        return 0


def _update_phase_tracker(tls_id: str):
    now = traci.simulation.getTime()
    try:
        phase = traci.trafficlight.getPhase(tls_id)
    except traci.TraCIException:
        return None

    info = _phase_tracker.get(tls_id)
    if info is None or info["phase"] != phase:
        _phase_tracker[tls_id] = {"phase": phase, "start_time": now}
    return _phase_tracker[tls_id]


def _elapsed_in_phase(tls_id: str) -> float:
    info = _update_phase_tracker(tls_id)
    if not info:
        return 0.0
    return traci.simulation.getTime() - info["start_time"]


def _remaining_phase_time(tls_id: str) -> float:
    try:
        return traci.trafficlight.getNextSwitch(tls_id) - traci.simulation.getTime()
    except traci.TraCIException:
        return 0.0


def _is_transition_state(state: str) -> bool:
    return any(c in "yYoO" for c in state)


def _has_green(state: str) -> bool:
    return any(c in "gG" for c in state)


def _incoming_lanes_for_green(tls_id: str, state: str) -> set:
    lanes = set()
    try:
        controlled_links = traci.trafficlight.getControlledLinks(tls_id)
    except traci.TraCIException:
        return lanes

    for idx, signal_char in enumerate(state):
        if signal_char not in "gG":
            continue
        if idx >= len(controlled_links):
            continue
        for link in controlled_links[idx]:
            if not link:
                continue
            in_lane = link[0]
            if in_lane and not in_lane.startswith(":"):
                lanes.add(in_lane)
    return lanes


def _all_incoming_lanes(tls_id: str) -> set:
    try:
        return {
            lane
            for lane in traci.trafficlight.getControlledLanes(tls_id)
            if lane and not lane.startswith(":")
        }
    except traci.TraCIException:
        return set()


def _queue_sum(lanes: set) -> int:
    total = 0
    for lane in lanes:
        try:
            total += traci.lane.getLastStepHaltingNumber(lane)
        except traci.TraCIException:
            pass
    return total


def _cooldown_ok(tls_id: str) -> bool:
    now = traci.simulation.getTime()
    last = _last_action_time.get(tls_id, -1e9)
    return (now - last) >= AS_ACTION_COOLDOWN


def _mark_action(tls_id: str):
    _last_action_time[tls_id] = traci.simulation.getTime()


def step(tl_ids: list[str]) -> str:
    """Run one adaptive signal control step and return active controller mode."""
    global _sorted_tl_ids
    _sorted_tl_ids = sorted(tl_ids)

    if config.USE_RL_SIGNALS and _rl_ready:
        sorted_ids = sorted(tl_ids)
        expected_obs_size = len(sorted_ids) * 6
        if expected_obs_size != TRAINED_OBS_SIZE:
            for tl_id in tl_ids:
                _phase_timers[tl_id] = _phase_timers.get(tl_id, 0) + 1
            return "Rule-Based (junction mismatch)"

        obs = _build_obs(sorted_ids)
        raw_action, _ = _dqn_model.predict(obs.reshape(1, -1), deterministic=True)
        decoded = [(int(raw_action) >> i) & 1 for i in range(len(sorted_ids))]

        for tl_id, action in zip(sorted_ids, decoded):
            if action == 1 and _phase_timers.get(tl_id, 0) >= config.MIN_GREEN_STEPS:
                try:
                    current = traci.trafficlight.getPhase(tl_id)
                    total = len(traci.trafficlight.getAllProgramLogics(tl_id)[0].phases)
                    traci.trafficlight.setPhase(tl_id, (current + 1) % total)
                    _phase_timers[tl_id] = 0
                except traci.TraCIException:
                    _phase_timers[tl_id] = _phase_timers.get(tl_id, 0) + 1
            else:
                _phase_timers[tl_id] = _phase_timers.get(tl_id, 0) + 1
        return "RL-DQN"

    if not ADAPTIVE_SIGNALS_ENABLED:
        for tls_id in tl_ids:
            _phase_timers[tls_id] = _phase_timers.get(tls_id, 0) + 1
        return "Rule-Based"

    for tls_id in traci.trafficlight.getIDList():
        try:
            state = traci.trafficlight.getRedYellowGreenState(tls_id)
        except traci.TraCIException:
            continue

        if not _has_green(state):
            continue
        if _is_transition_state(state):
            continue

        phase_count = _phase_count(tls_id)
        if phase_count <= 1:
            continue

        elapsed = _elapsed_in_phase(tls_id)
        remaining = _remaining_phase_time(tls_id)

        if elapsed < AS_MIN_GREEN:
            continue
        if not _cooldown_ok(tls_id):
            continue

        green_lanes = _incoming_lanes_for_green(tls_id, state)
        if not green_lanes:
            continue

        all_lanes = _all_incoming_lanes(tls_id)
        red_lanes = all_lanes - green_lanes

        green_q = _queue_sum(green_lanes)
        red_q = _queue_sum(red_lanes)

        # Case 1: Active demand on current green -> extend it a little
        if green_q >= AS_EXTEND_QUEUE and elapsed < AS_MAX_GREEN:
            try:
                if remaining < AS_EXTEND_STEP - 0.5:
                    traci.trafficlight.setPhaseDuration(tls_id, AS_EXTEND_STEP)
                    _mark_action(tls_id)
            except traci.TraCIException:
                pass
            continue

        # Case 2: Current green empty, others waiting -> move ahead
        if green_q <= AS_EMPTY_QUEUE and red_q >= AS_SWITCH_QUEUE:
            try:
                cur_phase = traci.trafficlight.getPhase(tls_id)
                next_phase = (cur_phase + 1) % phase_count
                traci.trafficlight.setPhase(tls_id, next_phase)
                _mark_action(tls_id)
            except traci.TraCIException:
                pass

    for tls_id in tl_ids:
        _phase_timers[tls_id] = _phase_timers.get(tls_id, 0) + 1
    return "Rule-Based"


def get_mode() -> str:
    """Return the currently selected control mode label."""
    return "RL-DQN" if (_rl_ready and config.USE_RL_SIGNALS) else "Rule-Based"


def reset_timers() -> None:
    """Reset phase timers at episode start."""
    _phase_timers.clear()
