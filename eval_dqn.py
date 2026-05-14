"""Evaluate trained DQN model and compare with rule-based baseline."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3 import DQN
import traci

import config
from rl_environments.signal_env import TrafficSignalEnv

EVAL_EPISODES = 3
MAX_STEPS_PER_EPISODE = 1_800
MODEL_PATH = os.path.join("outputs", "models", "dqn_signal_final")
EVAL_KPI_PATH = os.path.join("outputs", "dqn_eval_kpi.csv")
EVAL_SUMMARY_PATH = os.path.join("outputs", "dqn_eval_summary.json")
BASELINE_SUMMARY_PATH = os.path.join("outputs", "baseline_summary.csv")
RULE_BASED_MODE_NAME = "rule_based_v2x"


def _step_kpi(step: int) -> dict[str, float]:
    """Collect per-step KPI values from active vehicles."""
    vehicle_ids = traci.vehicle.getIDList()
    if not vehicle_ids:
        return {
            "step": float(step),
            "avg_speed_mps": 0.0,
            "avg_wait_s": 0.0,
            "total_wait_s": 0.0,
            "vehicle_count": 0.0,
        }

    speeds = [float(traci.vehicle.getSpeed(veh_id)) for veh_id in vehicle_ids]
    waits = [float(traci.vehicle.getWaitingTime(veh_id)) for veh_id in vehicle_ids]
    total_wait = float(sum(waits))
    return {
        "step": float(step),
        "avg_speed_mps": float(np.mean(speeds)),
        "avg_wait_s": float(np.mean(waits)),
        "total_wait_s": total_wait,
        "vehicle_count": float(len(vehicle_ids)),
    }


def _print_baseline_comparison(mean_speed: float, mean_wait: float) -> dict[str, float] | None:
    """Compare DQN summary with rule-based baseline if baseline summary exists."""
    if not os.path.exists(BASELINE_SUMMARY_PATH):
        print("Baseline summary not found; skipping baseline comparison.")
        return None

    baseline_df = pd.read_csv(BASELINE_SUMMARY_PATH)
    baseline_row = baseline_df[baseline_df["mode"] == RULE_BASED_MODE_NAME]
    if baseline_row.empty:
        print(f"Mode '{RULE_BASED_MODE_NAME}' not found in baseline summary.")
        return None

    base_speed = float(baseline_row.iloc[0]["avg_speed_mps"])
    base_wait = float(baseline_row.iloc[0]["avg_wait_s"])
    speed_improvement = 0.0 if base_speed == 0 else ((mean_speed - base_speed) / base_speed) * 100.0
    wait_improvement = 0.0 if base_wait == 0 else ((base_wait - mean_wait) / base_wait) * 100.0

    print(
        f"Speed improvement vs {RULE_BASED_MODE_NAME}: {speed_improvement:.2f}%"
    )
    print(
        f"Wait-time improvement vs {RULE_BASED_MODE_NAME}: {wait_improvement:.2f}%"
    )
    return {
        "speed_improvement_pct_vs_rule_based_v2x": speed_improvement,
        "wait_improvement_pct_vs_rule_based_v2x": wait_improvement,
    }


def main() -> None:
    """Run DQN evaluation episodes and persist KPI outputs."""
    os.makedirs("outputs", exist_ok=True)

    if not os.path.exists(f"{MODEL_PATH}.zip"):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}.zip")

    env = TrafficSignalEnv(sumo_cfg=config.SUMO_CFG, max_steps=MAX_STEPS_PER_EPISODE)
    model = DQN.load(MODEL_PATH)

    episode_rewards: list[float] = []
    all_steps: list[dict[str, float]] = []

    try:
        for episode in range(EVAL_EPISODES):
            obs, _ = env.reset()
            episode_reward = 0.0
            done = False
            step = 0

            while not done:
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _info = env.step(action)
                episode_reward += float(reward)
                row = _step_kpi(step)
                row["episode"] = float(episode)
                row["reward"] = float(reward)
                all_steps.append(row)
                step += 1
                done = bool(terminated or truncated)

            episode_rewards.append(episode_reward)
            print(f"Episode {episode + 1}/{EVAL_EPISODES} total reward: {episode_reward:.3f}")
    finally:
        env.close()

    kpi_df = pd.DataFrame(all_steps)
    kpi_df.to_csv(EVAL_KPI_PATH, index=False)
    print(f"Saved KPI: {EVAL_KPI_PATH}")

    mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    std_reward = float(np.std(episode_rewards)) if episode_rewards else 0.0
    mean_speed = float(kpi_df["avg_speed_mps"].mean()) if not kpi_df.empty else 0.0
    mean_wait = float(kpi_df["avg_wait_s"].mean()) if not kpi_df.empty else 0.0

    print(f"Episode total reward: {mean_reward:.3f} ± {std_reward:.3f}")
    print(f"Mean average speed: {mean_speed:.3f} m/s")
    print(f"Mean average waiting time: {mean_wait:.3f} s")

    summary: dict[str, Any] = {
        "episodes": EVAL_EPISODES,
        "mean_episode_total_reward": mean_reward,
        "std_episode_total_reward": std_reward,
        "mean_avg_speed_mps": mean_speed,
        "mean_avg_wait_s": mean_wait,
    }
    comparison = _print_baseline_comparison(mean_speed=mean_speed, mean_wait=mean_wait)
    if comparison is not None:
        summary.update(comparison)

    with open(EVAL_SUMMARY_PATH, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved summary: {EVAL_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
