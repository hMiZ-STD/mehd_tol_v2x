"""Gymnasium environment for DQN-based adaptive traffic signal control."""

from __future__ import annotations

import socket
from typing import Any

import gymnasium as gym
import numpy as np
import sumolib
import traci
from gymnasium import spaces

QUEUE_NORM_DENOMINATOR = 20.0
SPEED_NORM_DENOMINATOR = 13.89  # 50 km/h in m/s
PHASE_TIME_NORM_DENOMINATOR = 30.0
JUNCTION_FEATURES = 6
NS_ANGLE_RANGES = ((315.0, 360.0), (0.0, 45.0), (135.0, 225.0))


def find_free_port() -> int:
    """Return a free TCP port for launching TraCI without conflicts."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class TrafficSignalEnv(gym.Env[np.ndarray, np.ndarray]):
    """Traffic signal control environment for SUMO with multi-junction actions."""

    metadata = {"render_modes": []}

    def __init__(self, sumo_cfg: str, max_steps: int = 1800, min_green: int = 10) -> None:
        """Initialize the environment.

        Args:
            sumo_cfg: Path to SUMO configuration file.
            max_steps: Max simulation steps before truncation.
            min_green: Minimum phase duration before allowing phase advance.
        """
        super().__init__()
        self.sumo_cfg = sumo_cfg
        self.max_steps = max_steps
        self.min_green = min_green

        self.step_count = 0
        self.port: int | None = None
        self.tl_ids: list[str] = []
        self.incoming_lanes: dict[str, dict[str, list[str]]] = {}
        self.phase_timer: dict[str, int] = {}
        self.last_phase: dict[str, int] = {}
        self.num_junctions = 0

        self._initialize_spaces_from_network()

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        """Reset environment and start a fresh SUMO episode."""
        super().reset(seed=seed)
        del options

        self.close()
        self.step_count = 0

        binary = sumolib.checkBinary("sumo")
        self.port = find_free_port()
        traci.start(
            [binary, "-c", self.sumo_cfg, "--no-step-log", "--no-warnings"],
            port=self.port,
        )

        self.tl_ids = sorted(traci.trafficlight.getIDList())
        live_count = len(self.tl_ids)
        if live_count != self.num_junctions:
            raise RuntimeError(
                f"Traffic light count changed from {self.num_junctions} to {live_count}. "
                "Observation/action spaces must remain fixed for SB3."
            )
        self._build_lane_groups()
        self.phase_timer = {tl_id: 0 for tl_id in self.tl_ids}
        self.last_phase = {tl_id: self._safe_get_phase(tl_id) for tl_id in self.tl_ids}

        obs = self._get_obs()
        info: dict[str, Any] = {"num_junctions": len(self.tl_ids)}
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Apply action, advance simulation, and return transition tuple."""
        self._apply_action(self._decode_action(int(action)))
        traci.simulationStep()
        self.step_count += 1
        self._update_phase_timers()

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = False
        truncated = self.step_count >= self.max_steps
        info: dict[str, Any] = {"step_count": self.step_count}
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Build normalized observation vector across all junctions."""
        obs_values: list[float] = []
        for tl_id in self.tl_ids:
            lane_groups = self.incoming_lanes.get(tl_id, {"ns": [], "ew": []})
            ns_lanes = lane_groups["ns"]
            ew_lanes = lane_groups["ew"]

            queue_ns = self._sum_halting(ns_lanes) / QUEUE_NORM_DENOMINATOR
            queue_ew = self._sum_halting(ew_lanes) / QUEUE_NORM_DENOMINATOR
            avg_speed_ns = self._mean_speed(ns_lanes) / SPEED_NORM_DENOMINATOR
            avg_speed_ew = self._mean_speed(ew_lanes) / SPEED_NORM_DENOMINATOR

            phase = self._safe_get_phase(tl_id)
            total_phases = self._safe_get_total_phases(tl_id)
            phase_encoded = 0.0 if total_phases <= 0 else phase / float(total_phases)

            phase_time_norm = min(
                1.0,
                float(self.phase_timer.get(tl_id, 0)) / PHASE_TIME_NORM_DENOMINATOR,
            )

            obs_values.extend(
                [
                    float(np.clip(queue_ns, 0.0, 1.0)),
                    float(np.clip(queue_ew, 0.0, 1.0)),
                    float(np.clip(avg_speed_ns, 0.0, 1.0)),
                    float(np.clip(avg_speed_ew, 0.0, 1.0)),
                    float(np.clip(phase_encoded, 0.0, 1.0)),
                    float(np.clip(phase_time_norm, 0.0, 1.0)),
                ]
            )

        if not obs_values:
            return np.zeros((0,), dtype=np.float32)
        return np.asarray(obs_values, dtype=np.float32)

    def _compute_reward(self) -> float:
        """Return negative normalized total waiting time."""
        vehicle_ids = traci.vehicle.getIDList()
        if not vehicle_ids:
            return 0.0
        total_wait = sum(traci.vehicle.getWaitingTime(veh_id) for veh_id in vehicle_ids)
        return -float(total_wait) / 1000.0

    def _decode_action(self, action: int) -> list[int]:
        """Decode discrete action into per-junction binary actions."""
        return [(action >> i) & 1 for i in range(self.num_junctions)]

    def _apply_action(self, action: list[int]) -> None:
        """Apply per-junction keep/advance actions with minimum green protection."""
        if len(self.tl_ids) == 0:
            return

        if len(action) != len(self.tl_ids):
            raise ValueError(
                f"Action length {len(action)} does not match junction count {len(self.tl_ids)}"
            )

        for idx, tl_id in enumerate(self.tl_ids):
            requested = int(action[idx])
            if requested != 1:
                continue
            if self.phase_timer.get(tl_id, 0) < self.min_green:
                continue

            phase_count = self._safe_get_total_phases(tl_id)
            if phase_count <= 1:
                continue
            current_phase = self._safe_get_phase(tl_id)
            next_phase = (current_phase + 1) % phase_count
            traci.trafficlight.setPhase(tl_id, next_phase)

    def close(self) -> None:
        """Close active TraCI connection if present."""
        try:
            traci.close()
        except Exception:
            pass

    def render(self) -> None:
        """No-op render stub."""
        return

    def _build_lane_groups(self) -> None:
        """Build deterministic NS/EW incoming lane groups per traffic light."""
        self.incoming_lanes = {}
        for tl_id in self.tl_ids:
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
                    if self._is_ns_edge(edge_id):
                        ns_lanes.add(incoming_lane)
                    else:
                        ew_lanes.add(incoming_lane)

            self.incoming_lanes[tl_id] = {
                "ns": sorted(ns_lanes),
                "ew": sorted(ew_lanes),
            }

    @staticmethod
    def _is_ns_edge(edge_id: str) -> bool:
        """Classify edge direction by angle into NS or EW bucket."""
        try:
            angle = float(traci.edge.getAngle(edge_id))
        except traci.TraCIException:
            return False

        angle = angle % 360.0
        for low, high in NS_ANGLE_RANGES:
            if low <= angle <= high:
                return True
        return False

    @staticmethod
    def _sum_halting(lanes: list[str]) -> float:
        total = 0.0
        for lane_id in lanes:
            try:
                total += float(traci.lane.getLastStepHaltingNumber(lane_id))
            except traci.TraCIException:
                continue
        return total

    @staticmethod
    def _mean_speed(lanes: list[str]) -> float:
        if not lanes:
            return 0.0
        speeds: list[float] = []
        for lane_id in lanes:
            try:
                vehicle_count = int(traci.lane.getLastStepVehicleNumber(lane_id))
                if vehicle_count == 0:
                    speeds.append(0.0)
                else:
                    speeds.append(float(traci.lane.getLastStepMeanSpeed(lane_id)))
            except traci.TraCIException:
                continue
        if not speeds:
            return 0.0
        return float(np.mean(speeds))

    @staticmethod
    def _safe_get_phase(tl_id: str) -> int:
        try:
            return int(traci.trafficlight.getPhase(tl_id))
        except traci.TraCIException:
            return 0

    @staticmethod
    def _safe_get_total_phases(tl_id: str) -> int:
        try:
            logics = traci.trafficlight.getAllProgramLogics(tl_id)
            if not logics:
                return 1
            return max(1, int(len(logics[0].phases)))
        except traci.TraCIException:
            return 1

    def _update_phase_timers(self) -> None:
        """Update per-junction phase timers after each simulation step."""
        for tl_id in self.tl_ids:
            current_phase = self._safe_get_phase(tl_id)
            previous_phase = self.last_phase.get(tl_id, current_phase)
            if current_phase == previous_phase:
                self.phase_timer[tl_id] = self.phase_timer.get(tl_id, 0) + 1
            else:
                self.phase_timer[tl_id] = 0
            self.last_phase[tl_id] = current_phase

    def _initialize_spaces_from_network(self) -> None:
        """Probe network once to set fixed observation/action spaces for SB3."""
        binary = sumolib.checkBinary("sumo")
        probe_port = find_free_port()
        probe_cmd = [binary, "-c", self.sumo_cfg, "--no-step-log", "--no-warnings"]
        try:
            traci.start(probe_cmd, port=probe_port)
            self.tl_ids = sorted(traci.trafficlight.getIDList())
            self.num_junctions = len(self.tl_ids)
        finally:
            try:
                traci.close()
            except Exception:
                pass

        if self.num_junctions <= 0:
            raise RuntimeError("No traffic lights discovered in network; cannot build RL spaces.")

        self.action_space = spaces.Discrete(2 ** self.num_junctions)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_junctions * JUNCTION_FEATURES,),
            dtype=np.float32,
        )
