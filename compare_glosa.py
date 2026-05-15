"""Run and compare rule-based vs PPO-GLOSA scenarios."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd

import config
import main

RUN_STEPS = config.SIM_STEPS
OUTPUT_DIR = "outputs"


def _scenario_csv(name: str) -> str:
    return os.path.join(OUTPUT_DIR, f"{name}_kpi.csv")


def _run_scenarios() -> dict[str, str]:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    paths = {
        "scenario_a_rule_rule": _scenario_csv("scenario_a_rule_rule"),
        "scenario_b_rule_ppo_glosa": _scenario_csv("scenario_b_rule_ppo_glosa"),
        "scenario_c_dqn_ppo_glosa": _scenario_csv("scenario_c_dqn_ppo_glosa"),
    }

    main.run(
        no_gui=True,
        mode="rule_based_v2x",
        sim_steps=RUN_STEPS,
        output_csv=paths["scenario_a_rule_rule"],
        use_rl_signals=False,
        use_rl_glosa=False,
    )
    main.run(
        no_gui=True,
        mode="rule_based_v2x",
        sim_steps=RUN_STEPS,
        output_csv=paths["scenario_b_rule_ppo_glosa"],
        use_rl_signals=False,
        use_rl_glosa=True,
    )
    main.run(
        no_gui=True,
        mode="rule_based_v2x",
        sim_steps=RUN_STEPS,
        output_csv=paths["scenario_c_dqn_ppo_glosa"],
        use_rl_signals=True,
        use_rl_glosa=True,
    )
    return paths


def _summarize(path: str) -> dict[str, float]:
    df = pd.read_csv(path)
    avg_speed = float(df["avg_speed_mps"].mean()) if not df.empty else 0.0
    avg_wait = float(df["avg_wait_s"].mean()) if not df.empty else 0.0
    throughput = int(df["total_arrived_cum"].max()) if not df.empty else 0
    if {"co2_mg_s", "total_vehicles"}.issubset(df.columns):
        denom = df["total_vehicles"].replace(0, pd.NA)
        avg_co2 = float((df["co2_mg_s"] / denom).dropna().mean())
    else:
        avg_co2 = 0.0
    return {
        "avg_speed_mps": avg_speed,
        "avg_wait_s": avg_wait,
        "throughput": throughput,
        "avg_co2_per_vehicle_mg": avg_co2,
    }


def _print_table(summary_df: pd.DataFrame) -> None:
    print("\nGLOSA Comparison")
    print("=" * 94)
    print(
        f"{'Scenario':<34}"
        f"{'Avg Speed (m/s)':>16}"
        f"{'Avg Wait (s)':>14}"
        f"{'Throughput':>12}"
        f"{'Avg CO2/veh':>18}"
    )
    print("-" * 94)
    for _, row in summary_df.iterrows():
        print(
            f"{row['scenario']:<34}"
            f"{row['avg_speed_mps']:>16.3f}"
            f"{row['avg_wait_s']:>14.3f}"
            f"{int(row['throughput']):>12d}"
            f"{row['avg_co2_per_vehicle_mg']:>18.3f}"
        )
    print("=" * 94)


def _plot(summary_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    scenarios = summary_df["scenario"].tolist()
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    metrics = [
        ("avg_speed_mps", "Avg Speed (m/s)"),
        ("avg_wait_s", "Avg Wait (s)"),
        ("throughput", "Throughput"),
        ("avg_co2_per_vehicle_mg", "Avg CO2/veh (mg)"),
    ]
    for ax, (col, title) in zip(axes, metrics):
        ax.bar(scenarios, summary_df[col], color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "glosa_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def main_compare() -> None:
    """Run scenarios and create comparison outputs."""
    paths = _run_scenarios()
    rows = []
    for scenario, csv_path in paths.items():
        row = {"scenario": scenario}
        row.update(_summarize(csv_path))
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    _print_table(summary_df)
    out_csv = os.path.join(OUTPUT_DIR, "glosa_comparison.csv")
    summary_df.to_csv(out_csv, index=False)
    print(f"Saved summary: {out_csv}")
    _plot(summary_df)


if __name__ == "__main__":
    main_compare()
