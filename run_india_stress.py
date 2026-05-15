"""Run Indian mixed-traffic stress scenarios and summarize results."""

from __future__ import annotations

import os
import re
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

import config
import main

STRESS_CFG = "india_stress.sumocfg"
STEPS = 1800
OUTPUT_DIR = "outputs"
BASELINE_SUMMARY = os.path.join(OUTPUT_DIR, "baseline_summary.csv")


def _parse_collisions_from_log(log_path: str) -> float:
    """Parse collisions count from SUMO log/report output file."""
    if not os.path.exists(log_path):
        return 0.0
    text = open(log_path, "r", encoding="utf-8", errors="ignore").read()
    matches = re.findall(r"Collisions:\s*(\d+)", text)
    if not matches:
        return 0.0
    return float(matches[-1])


def _scenario_csv(name: str) -> str:
    return os.path.join(OUTPUT_DIR, f"india_{name}_kpi.csv")


def _run_one(
    name: str,
    mode: str,
    use_rl_signals: bool,
    use_rl_glosa: bool,
) -> tuple[str, float]:
    """Run one stress scenario and return output csv + collision count."""
    csv_path = _scenario_csv(name)
    original_cfg = config.SUMO_CFG
    try:
        config.SUMO_CFG = STRESS_CFG
        main.run(
            no_gui=True,
            mode=mode,
            sim_steps=STEPS,
            output_csv=csv_path,
            use_rl_signals=use_rl_signals,
            use_rl_glosa=use_rl_glosa,
        )
        collisions = _parse_collisions_from_log(os.path.join(config.LOG_DIR, "sumo.log"))
        return csv_path, collisions
    finally:
        config.SUMO_CFG = original_cfg


def _summarize(csv_path: str, collisions: float) -> dict[str, float]:
    df = pd.read_csv(csv_path)
    avg_speed = float(df["avg_speed_mps"].mean()) if not df.empty else 0.0
    avg_wait = float(df["avg_wait_s"].mean()) if not df.empty else 0.0
    throughput = int(df["total_arrived_cum"].max()) if not df.empty else 0
    collision_rate = collisions / max(1.0, float(throughput))
    return {
        "avg_speed_mps": avg_speed,
        "avg_wait_s": avg_wait,
        "throughput": throughput,
        "collision_rate": collision_rate,
        "collisions": collisions,
    }


def _load_normal_reference() -> dict[str, float]:
    if not os.path.exists(BASELINE_SUMMARY):
        return {"avg_speed_mps": 0.0, "avg_wait_s": 0.0, "throughput": 0.0, "collision_rate": 0.0}
    base_df = pd.read_csv(BASELINE_SUMMARY)
    row = base_df[base_df["mode"] == "rule_based_v2x"]
    if row.empty:
        return {"avg_speed_mps": 0.0, "avg_wait_s": 0.0, "throughput": 0.0, "collision_rate": 0.0}
    return {
        "avg_speed_mps": float(row.iloc[0]["avg_speed_mps"]),
        "avg_wait_s": float(row.iloc[0]["avg_wait_s"]),
        "throughput": float(row.iloc[0]["throughput"]),
        "collision_rate": 0.0,
    }


def _print_table(df: pd.DataFrame) -> None:
    print("\n=== Indian Traffic Stress Test Results ===")
    print("=" * 90)
    print(
        f"{'Scenario':<20}"
        f"{'Avg Speed':>12}"
        f"{'Avg Wait':>12}"
        f"{'Throughput':>12}"
        f"{'Collision Rate':>16}"
        f"{'Collisions':>12}"
    )
    print("-" * 90)
    for _, row in df.iterrows():
        print(
            f"{row['scenario']:<20}"
            f"{row['avg_speed_mps']:>12.3f}"
            f"{row['avg_wait_s']:>12.3f}"
            f"{int(row['throughput']):>12d}"
            f"{row['collision_rate']:>16.4f}"
            f"{int(row['collisions']):>12d}"
        )
    print("=" * 90)


def _plot(df: pd.DataFrame) -> None:
    metrics = [
        ("avg_speed_mps", "Avg Speed (m/s)"),
        ("avg_wait_s", "Avg Wait (s)"),
        ("throughput", "Throughput"),
        ("collision_rate", "Collision Rate"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    labels = df["scenario"].tolist()
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    for ax, (col, title) in zip(axes, metrics):
        ax.bar(labels, df[col], color=colors[: len(labels)])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "india_stress_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def main_stress() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    scenario_defs = [
        ("normal_traffic", None, None, None),
        ("india_null", "null_baseline", False, False),
        ("india_rule_based", "rule_based_v2x", False, False),
        ("india_full_rl", "rule_based_v2x", True, False),
    ]

    rows: list[dict[str, Any]] = []
    normal = _load_normal_reference()
    rows.append({"scenario": "normal_traffic", **normal, "collisions": 0.0})

    for name, mode, use_rl_signals, use_rl_glosa in scenario_defs[1:]:
        csv_path, collisions = _run_one(
            name=name,
            mode=mode,
            use_rl_signals=use_rl_signals,
            use_rl_glosa=use_rl_glosa,
        )
        summary = _summarize(csv_path, collisions)
        rows.append({"scenario": name, **summary})

    result_df = pd.DataFrame(rows)
    _print_table(result_df)

    out_csv = os.path.join(OUTPUT_DIR, "india_stress_results.csv")
    result_df.to_csv(out_csv, index=False)
    print(f"Saved results: {out_csv}")
    _plot(result_df)


if __name__ == "__main__":
    main_stress()
