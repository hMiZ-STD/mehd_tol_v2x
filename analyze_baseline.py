"""Analyze baseline benchmark outputs and generate comparison artifacts."""

from __future__ import annotations

import os
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

import config

MODES = ("null_baseline", "rule_based_v2x", "rule_based_no_ev")


def _csv_path(mode: str) -> str:
    return os.path.join(config.LOG_DIR, f"{mode}_kpi.csv")


def _safe_mean(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(series.mean())


def _mode_stats(mode: str, df: pd.DataFrame) -> Dict[str, float]:
    avg_speed = _safe_mean(df.get("avg_speed_mps", pd.Series(dtype=float)))
    avg_wait = _safe_mean(df.get("avg_wait_s", pd.Series(dtype=float)))
    p95_wait = float(df["avg_wait_s"].quantile(0.95)) if "avg_wait_s" in df and not df.empty else 0.0
    throughput = (
        int(df["total_arrived_cum"].max())
        if "total_arrived_cum" in df and not df.empty
        else 0
    )

    if {"co2_mg_s", "total_vehicles"}.issubset(df.columns):
        denom = df["total_vehicles"].replace(0, pd.NA)
        per_vehicle_co2 = (df["co2_mg_s"] / denom).dropna()
        avg_co2_per_vehicle = _safe_mean(per_vehicle_co2)
    else:
        avg_co2_per_vehicle = 0.0

    return {
        "mode": mode,
        "avg_speed_mps": avg_speed,
        "avg_wait_s": avg_wait,
        "throughput": throughput,
        "avg_co2_mg_per_vehicle_step": avg_co2_per_vehicle,
        "p95_wait_s": p95_wait,
    }


def _print_table(summary_df: pd.DataFrame) -> None:
    print("\nBaseline KPI Comparison")
    print("=" * 96)
    header = (
        f"{'Mode':<22}"
        f"{'Avg Speed (m/s)':>16}"
        f"{'Avg Wait (s)':>14}"
        f"{'Throughput':>12}"
        f"{'Avg CO2/veh (mg/step)':>24}"
        f"{'P95 Wait (s)':>14}"
    )
    print(header)
    print("-" * 96)
    for _, row in summary_df.iterrows():
        print(
            f"{row['mode']:<22}"
            f"{row['avg_speed_mps']:>16.3f}"
            f"{row['avg_wait_s']:>14.3f}"
            f"{int(row['throughput']):>12d}"
            f"{row['avg_co2_mg_per_vehicle_step']:>24.3f}"
            f"{row['p95_wait_s']:>14.3f}"
        )
    print("=" * 96)


def _plot_metric_grid(summary_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    metrics = [
        ("avg_speed_mps", "Average Speed (m/s)", "m/s"),
        ("avg_wait_s", "Average Wait Time (s)", "s"),
        ("avg_co2_mg_per_vehicle_step", "Avg CO2 per Vehicle (mg/step)", "mg/step"),
        ("p95_wait_s", "95th Percentile Wait Time (s)", "s"),
    ]
    modes = summary_df["mode"].tolist()

    for ax, (col, title, ylabel) in zip(axes, metrics):
        ax.bar(modes, summary_df[col], color=["#4C78A8", "#F58518", "#54A24B"])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    out_path = os.path.join(config.LOG_DIR, "baseline_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _plot_throughput(summary_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        summary_df["mode"],
        summary_df["throughput"],
        color=["#4C78A8", "#F58518", "#54A24B"],
    )
    ax.set_title("Throughput Comparison")
    ax.set_ylabel("Completed Trips (vehicles)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    out_path = os.path.join(config.LOG_DIR, "throughput_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    os.makedirs(config.LOG_DIR, exist_ok=True)

    stats_rows = []
    for mode in MODES:
        path = _csv_path(mode)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing KPI CSV for mode '{mode}': {path}. Run run_baseline.py first."
            )
        df = pd.read_csv(path)
        stats_rows.append(_mode_stats(mode, df))

    summary_df = pd.DataFrame(stats_rows)
    _print_table(summary_df)

    summary_path = os.path.join(config.LOG_DIR, "baseline_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    _plot_metric_grid(summary_df)
    _plot_throughput(summary_df)


if __name__ == "__main__":
    main()
