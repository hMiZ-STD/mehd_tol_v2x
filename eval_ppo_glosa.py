"""Evaluate PPO-GLOSA model and compare with rule-based baseline."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd
import traci
from stable_baselines3 import PPO

import config
from rl_environments.glosa_env import GLOSAEnv

EVAL_EPISODES = 3
MODEL_PATH = os.path.join("outputs", "models", "ppo_glosa_final")
BASELINE_PATH = os.path.join("outputs", "baseline_summary.csv")
SUMMARY_PATH = os.path.join("outputs", "ppo_glosa_eval_summary.json")
BASELINE_MODE = "rule_based_v2x"


def _episode_metrics(env: GLOSAEnv, model: PPO) -> dict[str, float]:
    """Run one eval episode and return summary metrics."""
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    step_speeds: list[float] = []
    step_waits: list[float] = []
    total_stops = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _info = env.step(action)
        total_reward += float(reward)

        vehicles = traci.vehicle.getIDList()
        if vehicles:
            speeds = [traci.vehicle.getSpeed(v) for v in vehicles]
            waits = [traci.vehicle.getWaitingTime(v) for v in vehicles]
            step_speeds.append(float(np.mean(speeds)))
            step_waits.append(float(np.mean(waits)))
            total_stops += float(sum(1 for speed in speeds if speed < 0.1))

        done = bool(terminated or truncated)

    return {
        "avg_speed": float(np.mean(step_speeds)) if step_speeds else 0.0,
        "avg_wait": float(np.mean(step_waits)) if step_waits else 0.0,
        "total_stops": total_stops,
        "total_reward": total_reward,
    }


def main() -> None:
    """Run PPO-GLOSA evaluation."""
    os.makedirs("outputs", exist_ok=True)
    if not os.path.exists(f"{MODEL_PATH}.zip"):
        raise FileNotFoundError(f"PPO-GLOSA model not found: {MODEL_PATH}.zip")

    env = GLOSAEnv(config.SUMO_CFG)
    model = PPO.load(MODEL_PATH)
    rows: list[dict[str, float]] = []

    try:
        for _ in range(EVAL_EPISODES):
            rows.append(_episode_metrics(env, model))
    finally:
        env.close()

    rewards = [row["total_reward"] for row in rows]
    avg_speeds = [row["avg_speed"] for row in rows]
    avg_waits = [row["avg_wait"] for row in rows]
    stops = [row["total_stops"] for row in rows]

    mean_reward = float(np.mean(rewards)) if rewards else 0.0
    std_reward = float(np.std(rewards)) if rewards else 0.0
    mean_speed = float(np.mean(avg_speeds)) if avg_speeds else 0.0
    mean_wait = float(np.mean(avg_waits)) if avg_waits else 0.0
    mean_stops = float(np.mean(stops)) if stops else 0.0

    print("=== PPO-GLOSA Evaluation Results ===")
    print(f"Mean Episode Reward:     {mean_reward:.3f} ± {std_reward:.3f}")
    print(f"Mean Avg Speed:          {mean_speed:.3f} m/s")
    print(f"Mean Avg Wait Time:      {mean_wait:.3f} s")
    print(f"Mean Total Stops:        {mean_stops:.1f}")

    summary: dict[str, Any] = {
        "episodes": EVAL_EPISODES,
        "mean_episode_reward": mean_reward,
        "std_episode_reward": std_reward,
        "mean_avg_speed_mps": mean_speed,
        "mean_avg_wait_s": mean_wait,
        "mean_total_stops": mean_stops,
    }

    print("\n=== Comparison vs Rule-Based GLOSA ===")
    if os.path.exists(BASELINE_PATH):
        baseline_df = pd.read_csv(BASELINE_PATH)
        baseline_row = baseline_df[baseline_df["mode"] == BASELINE_MODE]
        if not baseline_row.empty:
            base_speed = float(baseline_row.iloc[0]["avg_speed_mps"])
            base_wait = float(baseline_row.iloc[0]["avg_wait_s"])
            speed_improvement = 0.0 if base_speed == 0 else ((mean_speed - base_speed) / base_speed) * 100.0
            wait_reduction = 0.0 if base_wait == 0 else ((mean_wait - base_wait) / base_wait) * 100.0
            print(f"Speed improvement:       {speed_improvement:+.1f}%")
            print(f"Wait time reduction:     {wait_reduction:+.1f}%")
            summary["speed_improvement_pct_vs_rule_based_v2x"] = speed_improvement
            summary["wait_time_reduction_pct_vs_rule_based_v2x"] = wait_reduction
        else:
            print("rule_based_v2x row missing in baseline summary.")
    else:
        print("baseline_summary.csv not found.")

    with open(SUMMARY_PATH, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
