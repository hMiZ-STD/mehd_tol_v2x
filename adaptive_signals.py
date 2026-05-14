import traci
from config import (
    ADAPTIVE_SIGNALS_ENABLED,
    AS_MIN_GREEN,
    AS_MAX_GREEN,
    AS_EXTEND_STEP,
    AS_ACTION_COOLDOWN,
    AS_EMPTY_QUEUE,
    AS_SWITCH_QUEUE,
    AS_EXTEND_QUEUE,
)

_phase_tracker: dict = {}
_last_action_time: dict = {}

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
        return {lane for lane in traci.trafficlight.getControlledLanes(tls_id) if lane and not lane.startswith(":")}
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

def apply_adaptive_signals(protected_tls=None):
    if not ADAPTIVE_SIGNALS_ENABLED:
        return

    if protected_tls is None:
        protected_tls = set()

    for tls_id in traci.trafficlight.getIDList():
        if tls_id in protected_tls:
            continue

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

