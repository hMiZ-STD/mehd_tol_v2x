"""Post-analysis for Indian traffic stress test results."""

from __future__ import annotations

import os

import pandas as pd

RESULTS_CSV = os.path.join("outputs", "india_stress_results.csv")
BASELINE_SUMMARY = os.path.join("outputs", "baseline_summary.csv")


def _get_row(df: pd.DataFrame, scenario: str) -> pd.Series:
    row = df[df["scenario"] == scenario]
    if row.empty:
        raise ValueError(f"Missing scenario row: {scenario}")
    return row.iloc[0]


def main_analysis() -> None:
    if not os.path.exists(RESULTS_CSV):
        raise FileNotFoundError(f"Missing file: {RESULTS_CSV}. Run run_india_stress.py first.")

    df = pd.read_csv(RESULTS_CSV)
    normal = _get_row(df, "normal_traffic")
    india_null = _get_row(df, "india_null")
    india_rule = _get_row(df, "india_rule_based")
    india_rl = _get_row(df, "india_full_rl")

    normal_avg_speed = float(normal["avg_speed_mps"])
    india_rule_speed = float(india_rule["avg_speed_mps"])
    india_null_speed = float(india_null["avg_speed_mps"])
    india_rl_speed = float(india_rl["avg_speed_mps"])

    resilience = 0.0 if normal_avg_speed == 0 else (india_rule_speed / normal_avg_speed) * 100.0
    rl_gain = 0.0 if india_null_speed == 0 else ((india_rl_speed - india_null_speed) / india_null_speed) * 100.0
    degradation = 0.0 if normal_avg_speed == 0 else ((normal_avg_speed - india_rule_speed) / normal_avg_speed) * 100.0

    standard_gain = 0.0
    if os.path.exists(BASELINE_SUMMARY):
        base_df = pd.read_csv(BASELINE_SUMMARY)
        null_row = base_df[base_df["mode"] == "null_baseline"]
        rule_row = base_df[base_df["mode"] == "rule_based_v2x"]
        if not null_row.empty and not rule_row.empty:
            b_null = float(null_row.iloc[0]["avg_speed_mps"])
            b_rule = float(rule_row.iloc[0]["avg_speed_mps"])
            standard_gain = 0.0 if b_null == 0 else ((b_rule - b_null) / b_null) * 100.0

    print(f"Rule-based V2X resilience under Indian traffic: {resilience:.1f}%")
    print(f"RL signal improvement under Indian traffic: {rl_gain:+.1f}%")
    print(f"Performance degradation vs normal traffic: {degradation:.1f}%")
    print(
        "Under Hyderabad Indian mixed-traffic conditions (40% two-wheelers, "
        "20% auto-rickshaws, 15% heavy two-wheelers), the V2X system maintained "
        f"{resilience:.1f}% of its normal-traffic performance. The DQN signal controller "
        f"achieved {rl_gain:+.1f}% speed improvement over no-control baseline under "
        f"Indian traffic, compared to {standard_gain:.1f}% in standard conditions."
    )


if __name__ == "__main__":
    main_analysis()
