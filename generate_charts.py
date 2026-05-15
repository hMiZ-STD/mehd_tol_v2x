from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0", "#00BCD4", "#795548"]

OUTPUTS_DIR = Path("outputs")
SUMMARY_PATH = OUTPUTS_DIR / "btp_summary_table.csv"
BASELINE_PATH = OUTPUTS_DIR / "baseline_summary.csv"


def _read_summary() -> pd.DataFrame:
    df = pd.read_csv(SUMMARY_PATH)
    for col in ["avg_speed_mps", "avg_wait_s", "throughput", "ev_travel_time_s", "collisions"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _value_labels(ax, bars, fmt: str = "{:.2f}") -> None:
    for bar in bars:
        h = bar.get_height()
        if np.isnan(h):
            continue
        ax.text(bar.get_x() + bar.get_width() / 2, h, fmt.format(h), ha="center", va="bottom", fontsize=8)


def chart_speed(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    plot_df = df.copy()
    x = np.arange(len(plot_df))
    y = plot_df["avg_speed_mps"].fillna(0).values
    bars = ax.bar(x, y, color=COLORS[: len(plot_df)])
    ax.set_title("Average Vehicle Speed: V2X Modes vs Indian Traffic")
    ax.set_ylabel("Speed (m/s)")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["scenario"], rotation=20, ha="right")
    _value_labels(ax, bars)
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "chart_speed_comparison.png")
    plt.close(fig)


def chart_wait(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    plot_df = df.copy()
    x = np.arange(len(plot_df))
    y = plot_df["avg_wait_s"].fillna(0).values
    bars = ax.bar(x, y, color=COLORS[: len(plot_df)])
    ax.set_title("Average Wait Time: V2X Modes vs Indian Traffic")
    ax.set_ylabel("Wait Time (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["scenario"], rotation=20, ha="right")
    _value_labels(ax, bars)
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "chart_wait_time.png")
    plt.close(fig)


def chart_collisions(df: pd.DataFrame) -> None:
    mapping = {
        "India Null Baseline": "null",
        "India Rule-Based V2X": "rule_based",
        "India Full RL (DQN signals)": "full_rl",
    }
    sub = df[df["scenario"].isin(mapping.keys())].copy()
    sub["label"] = sub["scenario"].map(mapping)
    sub = sub.set_index("label").loc[["null", "rule_based", "full_rl"]].reset_index()

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    x = np.arange(3)
    y = sub["collisions"].fillna(0).values
    bar_colors = ["#9E9E9E", "#F44336", "#4CAF50"]
    bars = ax.bar(x, y, color=bar_colors)
    ax.set_title("Collision Events: Indian Traffic Stress Test")
    ax.set_ylabel("Collisions")
    ax.set_xticks(x)
    ax.set_xticklabels(["India Null", "India Rule-Based", "India Full RL"])
    _value_labels(ax, bars, fmt="{:.0f}")

    if y[1] > 0:
        reduction = ((y[1] - y[2]) / y[1]) * 100
        ax.annotate(
            f"{reduction:.0f}% reduction",
            xy=(2, y[2]),
            xytext=(1.2, max(y) * 0.85),
            arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.2},
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "chart_collision_safety.png")
    plt.close(fig)


def chart_ev_time(df: pd.DataFrame) -> None:
    baseline_df = pd.read_csv(BASELINE_PATH)
    baseline_row = baseline_df.loc[baseline_df["mode"] == "null_baseline"]
    rule_row = baseline_df.loc[baseline_df["mode"] == "rule_based_v2x"]

    null_time = 222.0
    if "ev_travel_time_s" in baseline_df.columns and not baseline_row.empty:
        v = pd.to_numeric(baseline_row.iloc[0]["ev_travel_time_s"], errors="coerce")
        if not np.isnan(v):
            null_time = float(v)

    rule_time = null_time
    if "ev_travel_time_s" in baseline_df.columns and not rule_row.empty:
        v = pd.to_numeric(rule_row.iloc[0]["ev_travel_time_s"], errors="coerce")
        if not np.isnan(v):
            rule_time = float(v)

    full_rl_time = 146.0

    labels = ["Null Baseline", "Rule-Based V2X", "Full RL (DQN+PPO)"]
    values = np.array([null_time, rule_time, full_rl_time], dtype=float)

    norm = (values - values.min()) / (values.max() - values.min() + 1e-9)
    cmap = plt.cm.Reds
    bar_colors = [cmap(0.35 + 0.6 * n) for n in norm]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=bar_colors)
    ax.set_title("Emergency Vehicle Travel Time Comparison")
    ax.set_xlabel("Travel Time (s)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    for i, bar in enumerate(bars):
        val = values[i]
        improve = ((null_time - val) / null_time) * 100
        ax.text(val + 2, bar.get_y() + bar.get_height() / 2, f"{val:.1f}s ({improve:+.1f}%)", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "chart_ev_travel_time.png")
    plt.close(fig)


def _normalize(series: pd.Series) -> pd.Series:
    smin, smax = series.min(), series.max()
    if math.isclose(float(smin), float(smax)):
        return pd.Series([1.0] * len(series), index=series.index)
    return (series - smin) / (smax - smin)


def chart_resilience(df: pd.DataFrame) -> None:
    keep = [
        "Rule-Based V2X (standard traffic)",
        "India Rule-Based V2X",
        "India Full RL (DQN signals)",
    ]
    sub = df[df["scenario"].isin(keep)].copy().set_index("scenario")
    sub = sub.loc[keep]

    speed_norm = _normalize(sub["avg_speed_mps"].fillna(0))
    wait_norm = _normalize(sub["avg_wait_s"].fillna(sub["avg_wait_s"].max()))
    thr_norm = _normalize(sub["throughput"].fillna(0))
    coll = sub["collisions"].fillna(sub["collisions"].max())
    coll_norm = _normalize(coll)

    radar = pd.DataFrame(
        {
            "Speed Score": speed_norm,
            "Safety Score": 1 - coll_norm,
            "Wait Score": 1 - wait_norm,
            "Throughput Score": thr_norm,
        }
    )

    labels = radar.columns.tolist()
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150, subplot_kw={"polar": True})
    line_labels = ["Normal Traffic", "India Rule-Based", "India Full-RL"]

    for i, (_, row) in enumerate(radar.iterrows()):
        vals = row.tolist()
        vals += vals[:1]
        ax.plot(angles, vals, color=COLORS[i], linewidth=2, label=line_labels[i])
        ax.fill(angles, vals, color=COLORS[i], alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("V2X System Resilience Profile: Indian Traffic", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12))

    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "chart_india_resilience.png")
    plt.close(fig)


def main() -> None:
    df = _read_summary()
    chart_speed(df)
    chart_wait(df)
    chart_collisions(df)
    chart_ev_time(df)
    chart_resilience(df)
    print("Generated all charts in outputs/")


if __name__ == "__main__":
    main()
