"""Gymnasium environment for PPO-based fleet-level GLOSA control."""

from __future__ import annotations

import socket
from typing import Any

import gymnasium as gym
import numpy as np
import sumolib
import traci
from gymnasium import spaces

import config

OBS_DIM = 5
TIME_TO_SWITCH_NORM = 60.0
QUEUE_AHEAD_NORM = 10.0


def find_free_port() -> int:
    """Return a free localhost TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class GLOSAEnv(gym.Env[np.ndarray, np.ndarray]):
    """Fleet-level GLOSA environment controlling all non-EV vehicles each step."""

    metadata = {"render_modes": []}

    def __init__(self, sumo_cfg: str, max_steps: int = 1800) -> None:
        """Initialize environment.

        Args:
            sumo_cfg: Path to SUMO config file.
            max_steps: Episode length in simulation steps.
        """
        super().__init__()
        self.sumo_cfg = sumo_cfg
        self.max_steps = max_steps
        self.step_count = 0
        self.port: int | None = None

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict]:
        """Reset episode and launch SUMO."""
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
        return self._get_fleet_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Apply fleet action, advance sim by one step, and return transition."""
        self._apply_fleet_action(action)
        traci.simulationStep()
        self.step_count += 1

        obs = self._get_fleet_obs()
        reward = self._compute_reward()
        terminated = False
        truncated = self.step_count >= self.max_steps
        info = {"step_count": self.step_count}
        return obs, reward, terminated, truncated, info

    def _get_fleet_obs(self) -> np.ndarray:
        """Return fleet-mean normalized observation vector."""
        features: list[list[float]] = []
        for veh_id in traci.vehicle.getIDList():
            try:
                if traci.vehicle.getTypeID(veh_id) == config.EV_TYPE_ID:
                    continue
            except traci.TraCIException:
                continue

            tl = self._next_tl_info(veh_id)
            speed = float(traci.vehicle.getSpeed(veh_id))
            speed_norm = float(np.clip(speed / config.GLOSA_MAX_SPEED, 0.0, 1.0))

            if tl is None:
                features.append([0.0, speed_norm, 0.0, 0.0, 0.0])
            else:
                dist_norm = float(
                    np.clip(tl["dist"] / config.GLOSA_LOOKAHEAD, 0.0, 1.0)
                )
                tts_norm = float(
                    np.clip(tl["time_to_switch"] / TIME_TO_SWITCH_NORM, 0.0, 1.0)
                )
                queue_norm = float(
                    np.clip(tl["queue_ahead"] / QUEUE_AHEAD_NORM, 0.0, 1.0)
                )
                features.append(
                    [dist_norm, speed_norm, tts_norm, float(tl["is_green"]), queue_norm]
                )

        if not features:
            return np.zeros((OBS_DIM,), dtype=np.float32)

        fleet_mean = np.mean(np.asarray(features, dtype=np.float32), axis=0)
        return fleet_mean.astype(np.float32)

    def _compute_reward(self) -> float:
        """Compute fleet-level reward."""
        vehicles = traci.vehicle.getIDList()
        if not vehicles:
            return 0.0

        waits = [traci.vehicle.getWaitingTime(v) for v in vehicles]
        speeds = [traci.vehicle.getSpeed(v) for v in vehicles]
        avg_wait = float(np.mean(waits))
        avg_speed = float(np.mean(speeds))
        stops = sum(1 for speed in speeds if speed < 0.1)

        return (
            (avg_speed / config.GLOSA_MAX_SPEED)
            - (avg_wait / 30.0)
            - (stops / len(vehicles)) * 0.5
        )

    def _apply_fleet_action(self, action: np.ndarray) -> None:
        """Apply one fleet-wide speed multiplier to all non-EV vehicles."""
        multiplier = float(np.asarray(action).reshape(-1)[0])
        advisory_speed = float(np.clip(multiplier, 0.0, 1.0) * config.GLOSA_MAX_SPEED)
        advisory_speed = max(config.GLOSA_MIN_SPEED, min(config.GLOSA_MAX_SPEED, advisory_speed))

        for veh_id in traci.vehicle.getIDList():
            try:
                if traci.vehicle.getTypeID(veh_id) == config.EV_TYPE_ID:
                    continue
            except traci.TraCIException:
                continue

            tl = self._next_tl_info(veh_id)
            try:
                if tl is None:
                    traci.vehicle.setSpeed(veh_id, -1)
                else:
                    traci.vehicle.setSpeed(veh_id, advisory_speed)
            except traci.TraCIException:
                continue

    def _next_tl_info(self, veh_id: str) -> dict[str, float] | None:
        """Return next-TLS info for one vehicle or None."""
        try:
            tls_info = traci.vehicle.getNextTLS(veh_id)
            if not tls_info:
                return None
            tls_id, lane_idx, dist, state = tls_info[0]
            if dist > config.GLOSA_LOOKAHEAD or dist < 0.0:
                return None

            time_to_switch = (
                traci.trafficlight.getNextSwitch(tls_id) - traci.simulation.getTime()
            )
            lane_id = traci.vehicle.getLaneID(veh_id)
            queue_ahead = float(traci.lane.getLastStepHaltingNumber(lane_id))
            is_green = 1.0 if state in "gGuU" else 0.0

            del lane_idx
            return {
                "dist": float(dist),
                "time_to_switch": float(max(0.0, time_to_switch)),
                "is_green": float(is_green),
                "queue_ahead": float(queue_ahead),
            }
        except traci.TraCIException:
            return None

    def close(self) -> None:
        """Close TraCI connection if active."""
        try:
            traci.close()
        except Exception:
            pass

    def render(self) -> None:
        """No-op render."""
        return
