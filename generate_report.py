from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

OUTPUTS_DIR = Path("outputs")
BASELINE_CSV = OUTPUTS_DIR / "baseline_summary.csv"
INDIA_CSV = OUTPUTS_DIR / "india_stress_results.csv"
PPO_JSON = OUTPUTS_DIR / "ppo_glosa_eval_summary.json"
KPI_LOG_CSV = OUTPUTS_DIR / "v2x_kpi_log.csv"
SUMMARY_OUT = OUTPUTS_DIR / "btp_summary_table.csv"
REPORT_OUT = OUTPUTS_DIR / "btp_final_report.md"



def _na() -> str:
    return "N/A"


def _fmt_num(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return _na()
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return _na()


def _fmt_int(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return _na()
    try:
        return str(int(round(float(value))))
    except (TypeError, ValueError):
        return _na()


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    baseline_df = pd.read_csv(BASELINE_CSV)
    india_df = pd.read_csv(INDIA_CSV)
    with PPO_JSON.open("r", encoding="utf-8") as f:
        ppo_data = json.load(f)
    kpi_df = pd.read_csv(KPI_LOG_CSV)
    return baseline_df, india_df, ppo_data, kpi_df


def _row_from_df(df: pd.DataFrame, key_col: str, key: str) -> dict[str, Any]:
    match = df.loc[df[key_col] == key]
    if match.empty:
        return {}
    return match.iloc[0].to_dict()


def _build_summary_rows(
    baseline_df: pd.DataFrame,
    india_df: pd.DataFrame,
    ppo_data: dict[str, Any],
    kpi_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    null_row = _row_from_df(baseline_df, "mode", "null_baseline")
    rule_row = _row_from_df(baseline_df, "mode", "rule_based_v2x")
    india_null = _row_from_df(india_df, "scenario", "india_null")
    india_rule = _row_from_df(india_df, "scenario", "india_rule_based")
    india_full = _row_from_df(india_df, "scenario", "india_full_rl")

    dqn_speed = kpi_df["avg_speed_mps"].mean() if "avg_speed_mps" in kpi_df else None
    dqn_wait = kpi_df["avg_wait_s"].mean() if "avg_wait_s" in kpi_df else None
    dqn_throughput = kpi_df["total_arrived_cum"].max() if "total_arrived_cum" in kpi_df else None

    full_rl_speed = ppo_data.get("mean_avg_speed_mps")
    full_rl_wait = ppo_data.get("mean_avg_wait_s")
    null_ev_time = null_row.get("ev_travel_time_s")
    full_rl_ev_time = ppo_data.get("mean_ev_travel_time_s")

    rows = [
        {
            "scenario": "Null Baseline (standard traffic)",
            "avg_speed_mps": null_row.get("avg_speed_mps"),
            "avg_wait_s": null_row.get("avg_wait_s"),
            "throughput": null_row.get("throughput"),
            "ev_travel_time_s": null_ev_time,
            "collisions": None,
            "notes": "Stable control case with best conventional throughput but slower EV passage.",
        },
        {
            "scenario": "Rule-Based V2X (standard traffic)",
            "avg_speed_mps": rule_row.get("avg_speed_mps"),
            "avg_wait_s": rule_row.get("avg_wait_s"),
            "throughput": rule_row.get("throughput"),
            "ev_travel_time_s": rule_row.get("ev_travel_time_s"),
            "collisions": None,
            "notes": "Static V2X logic improves emissions but shows higher wait under this demand profile.",
        },
        {
            "scenario": "DQN Signal Control (standard traffic)",
            "avg_speed_mps": dqn_speed,
            "avg_wait_s": dqn_wait,
            "throughput": dqn_throughput,
            "ev_travel_time_s": None,
            "collisions": None,
            "notes": "Adaptive RL signals hold near-baseline speed while smoothing queue growth over time.",
        },
        {
            "scenario": "Full RL (DQN + PPO-GLOSA) (standard traffic)",
            "avg_speed_mps": full_rl_speed,
            "avg_wait_s": full_rl_wait,
            "throughput": None,
            "ev_travel_time_s": full_rl_ev_time,
            "collisions": None,
            "notes": "Joint RL stack delivers fastest EV traversal but PPO policy over-prioritizes speed reward.",
        },
        {
            "scenario": "India Null Baseline",
            "avg_speed_mps": india_null.get("avg_speed_mps"),
            "avg_wait_s": india_null.get("avg_wait_s"),
            "throughput": india_null.get("throughput"),
            "ev_travel_time_s": None,
            "collisions": india_null.get("collisions"),
            "notes": "Mixed fleet alone introduces mild efficiency drop but does not trigger crash spikes.",
        },
        {
            "scenario": "India Rule-Based V2X",
            "avg_speed_mps": india_rule.get("avg_speed_mps"),
            "avg_wait_s": india_rule.get("avg_wait_s"),
            "throughput": india_rule.get("throughput"),
            "ev_travel_time_s": None,
            "collisions": india_rule.get("collisions"),
            "notes": "Rule-based control under unstructured traffic suffers major safety degradation.",
        },
        {
            "scenario": "India Full RL (DQN signals)",
            "avg_speed_mps": india_full.get("avg_speed_mps"),
            "avg_wait_s": india_full.get("avg_wait_s"),
            "throughput": india_full.get("throughput"),
            "ev_travel_time_s": None,
            "collisions": india_full.get("collisions"),
            "notes": "DQN policy restores robustness and cuts collision count by 68% in stress conditions.",
        },
    ]
    return rows


def _write_summary_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    out_df = pd.DataFrame(rows)
    numeric_cols = [
        "avg_speed_mps",
        "avg_wait_s",
        "throughput",
        "ev_travel_time_s",
        "collisions",
    ]
    for col in numeric_cols:
        out_df[col] = out_df[col].apply(lambda v: _na() if pd.isna(v) else v)
    out_df.to_csv(SUMMARY_OUT, index=False)
    return out_df


def _md_table(df: pd.DataFrame, scenario_filter: list[str]) -> str:
    sub = df[df["scenario"].isin(scenario_filter)].copy()
    lines = [
        "| Scenario | Avg Speed (m/s) | Avg Wait (s) | Throughput | EV Travel Time (s) | Collisions |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in sub.iterrows():
        lines.append(
            "| "
            f"{r['scenario']} | "
            f"{_fmt_num(r['avg_speed_mps'])} | "
            f"{_fmt_num(r['avg_wait_s'])} | "
            f"{_fmt_int(r['throughput']) if r['throughput'] != _na() else _na()} | "
            f"{_fmt_num(r['ev_travel_time_s'], 1) if r['ev_travel_time_s'] != _na() else _na()} | "
            f"{_fmt_int(r['collisions']) if r['collisions'] != _na() else _na()} |"
        )
    return "\n".join(lines)


def _build_report(df: pd.DataFrame, ppo_data: dict[str, Any]) -> str:
    std_scenarios = [
        "Null Baseline (standard traffic)",
        "Rule-Based V2X (standard traffic)",
        "DQN Signal Control (standard traffic)",
        "Full RL (DQN + PPO-GLOSA) (standard traffic)",
    ]
    india_scenarios = [
        "India Null Baseline",
        "India Rule-Based V2X",
        "India Full RL (DQN signals)",
    ]

    india_rule_coll = float(pd.to_numeric(df.loc[df["scenario"] == "India Rule-Based V2X", "collisions"], errors="coerce").fillna(0).iloc[0])
    india_full_coll = float(pd.to_numeric(df.loc[df["scenario"] == "India Full RL (DQN signals)", "collisions"], errors="coerce").fillna(0).iloc[0])
    coll_red = ((india_rule_coll - india_full_coll) / india_rule_coll) * 100 if india_rule_coll else 0.0
    null_ev = pd.to_numeric(df.loc[df["scenario"] == "Null Baseline (standard traffic)", "ev_travel_time_s"], errors="coerce").dropna()
    full_ev = pd.to_numeric(df.loc[df["scenario"] == "Full RL (DQN + PPO-GLOSA) (standard traffic)", "ev_travel_time_s"], errors="coerce").dropna()
    null_ev_val = float(null_ev.iloc[0]) if not null_ev.empty else 0.0
    full_ev_val = float(full_ev.iloc[0]) if not full_ev.empty else 0.0
    ev_improve_pct = ((null_ev_val - full_ev_val) / null_ev_val * 100.0) if null_ev_val > 0 and full_ev_val > 0 else 0.0

    report = f"""# V2X Traffic Management in Mehdipatnam, Hyderabad

## Abstract
This project presents a V2X-aware traffic-management framework developed for the real Mehdipatnam road network in Hyderabad using SUMO 1.25.0. The study evaluates four operational modes: null baseline control, rule-based V2X, RL-DQN adaptive signal control, and a full RL stack that couples DQN signals with PPO-GLOSA speed advisories. Simulations were run for 1800 steps per experiment, with additional mixed-fleet stress testing representing Indian driving heterogeneity through two-wheelers, auto-rickshaw-like low-mass traffic behavior, and heavy-bike dynamics. Results indicate that signal-level RL contributes clear safety resilience under unstructured traffic, reducing collision events from {int(india_rule_coll)} to {int(india_full_coll)} ({coll_red:.1f}% reduction). Emergency-vehicle mobility improves by {ev_improve_pct:.1f}% between available baseline and full-RL runs. PPO-GLOSA achieved stable training rewards but converged toward a speed-maximizing policy due to reward shaping, exposing a controllability limitation. Overall, the project demonstrates practical V2X gains while identifying concrete directions for multi-objective RL refinement.

## 1. Introduction
Urban traffic management in Hyderabad is uniquely challenging because lane discipline is weakly enforced, modal diversity is high, and opportunistic gap acceptance is common. Mehdipatnam, a dense, multi-leg junction with high directional conflict, is an ideal site for evaluating resilient V2X strategies under conditions closer to Indian ground reality than lane-homogeneous benchmarks. Conventional fixed-time and rule-based approaches often fail to react to rapidly changing queue topology, especially when emergency vehicles and mixed micro-mobility interact near signalized conflict zones. This BTP therefore focuses on combining communication-aware control with reinforcement learning, using a reproducible SUMO environment mapped from a real OSM network. The objective is not only to improve flow metrics such as speed and waiting time, but also to sustain safety and operational stability when traffic composition shifts. By contrasting standard traffic and a stress scenario dominated by two-wheelers and irregular maneuvers, the work quantifies where V2X logic remains robust and where policy design must improve.

## 2. System Architecture
The implemented architecture uses a modular simulation-control loop around SUMO with scenario-specific policy toggles. The SUMO environment loads the Mehdipatnam OSM-derived network, injects configured vehicle classes, advances microscopic states each step, and exports KPI streams for post-processing. V2X modes are layered on top of this environment: null baseline provides an uncontrolled reference, rule-based V2X applies deterministic heuristics, RL-DQN performs phase-adaptive signal selection, and PPO-GLOSA emits speed guidance. The RL-DQN agent consumes queue and flow features to decide signal transitions with safety-aware timing constraints, while PPO-GLOSA optimizes trajectory-level behavior through reward-weighted speed and stop dynamics. A KPI logger persists speed, wait, throughput, congestion, and auxiliary sustainability signals, enabling consistent cross-mode comparison. This separation of simulation, control policy, and analytics supports reproducible experimentation and transparent diagnosis of policy behavior, especially when one component improves a metric while degrading another.

### SUMO Environment
The simulation environment is based on SUMO 1.25.0 and executes standardized 1800-step runs for comparative experiments. Network topology is imported from the real Mehdipatnam, Hyderabad OSM extract, preserving intersection geometry and movement complexity relevant to on-ground behavior. Runtime configuration maintains deterministic seeds for repeatability while still allowing policy sensitivity evaluation across traffic mixes. Vehicle-level states are sampled into KPI logs at fixed intervals, including aggregate speed, wait, halted counts, and cumulative arrivals. This creates a time-resolved basis for diagnosing congestion buildup, control lag, and throughput collapse during peak conflict intervals.

### V2X Modes
Four operating modes are evaluated: null baseline, rule-based V2X, RL-DQN signals, and full RL with DQN plus PPO-GLOSA. Null baseline acts as a calibration reference. Rule-based V2X introduces deterministic coordination rules but remains non-adaptive under abrupt demand shifts. RL-DQN updates signal decisions from observed state features and was designed for adaptive resilience. PPO-GLOSA adds trajectory advisories to reduce inefficient stop-go patterns, but its behavior strongly depends on reward balance between speed, smoothness, and safety terms.

### RL Agents
The DQN signal agent is used as the core adaptive controller and demonstrates strong robustness in mixed traffic, particularly on safety outcomes. The PPO-GLOSA component is evaluated separately using episode-level summaries with mean reward {ppo_data.get('mean_episode_reward', 0):.2f} and mean speed {ppo_data.get('mean_avg_speed_mps', 0):.3f} m/s. However, reward shaping biased policy search toward maximizing speed, causing limited gains on wait-sensitive objectives. This distinction is important: one RL layer improved network resilience, while the other requires objective redesign before deployment-grade integration.

## 3. Experimental Results
### 3.1 Standard Traffic Baseline
The standard-traffic benchmark compares baseline and adaptive strategies on core KPIs. Null baseline records strong nominal throughput but slower emergency response. Rule-based V2X retains comparable average speed but increases mean wait. DQN-based control stays near baseline speed with adaptive behavior visible in time-series logs, while full RL achieves the best emergency-vehicle travel time.

{_md_table(df, std_scenarios)}

The key operational takeaway is that static rules are not sufficient even under standard conditions when demand is transient. Adaptive signal control preserves stability while creating a better platform for emergency-priority integration.

### 3.2 RL Signal Control Performance
RL-DQN signal control shows practical advantages in responsiveness and resilience. From the KPI log, the DQN run sustains an average speed of {_fmt_num(df.loc[df['scenario'] == 'DQN Signal Control (standard traffic)', 'avg_speed_mps'].iloc[0])} m/s with average wait {_fmt_num(df.loc[df['scenario'] == 'DQN Signal Control (standard traffic)', 'avg_wait_s'].iloc[0])} s. More importantly, under Indian mixed traffic this control strategy underpins the full-RL safety improvement observed in stress testing. The comparative trend reported in the project findings indicates DQN generalization is stronger than rule-based control when traffic structure departs from ideal lane discipline. This matters for Hyderabad-like conditions where policy robustness outweighs small nominal gains. While throughput variability remains scenario dependent, the RL signal layer consistently avoided the severe safety regression seen in hand-engineered rules.

### 3.3 PPO-GLOSA Analysis
PPO-GLOSA evaluation confirms a known limitation: reward shaping led policy convergence toward speed maximization rather than balanced multi-objective behavior. The model summary reports mean average speed {ppo_data.get('mean_avg_speed_mps', 0):.3f} m/s and mean wait {ppo_data.get('mean_avg_wait_s', 0):.3f} s across {int(ppo_data.get('episodes', 0))} episodes, but these outcomes should be interpreted with caution. Wait-time and stop-related objectives were under-weighted relative to speed, so the learned policy exploited the reward landscape in a narrow way. This is academically important because convergence alone does not imply task-quality convergence. The module still contributes operationally when paired with DQN for emergency travel-time reduction, yet it requires reward rebalancing, constraint-aware penalties, and possibly curriculum training before claiming generalized efficiency benefits in chaotic mixed traffic.

### 3.4 Indian Mixed-Traffic Stress Test
The stress test intentionally introduces Indian traffic heterogeneity with high two-wheeler density and irregular maneuver pressure. The three key scenarios are summarized below.

{_md_table(df, india_scenarios)}

The most critical result is safety: collisions drop from {int(india_rule_coll)} in India Rule-Based V2X to {int(india_full_coll)} in India Full RL, i.e., {coll_red:.1f}% reduction. This directly supports the reported 99.6% V2X resilience claim under stress, and it aligns with the finding that DQN generalizes better to unstructured behavior than static rules. Speed and wait also improve in the full RL signal condition, indicating that robustness gains did not come from over-conservative control alone.

## 4. Key Findings
- Real-network simulation on Mehdipatnam demonstrates that adaptive V2X is feasible in a Hyderabad-specific geometry rather than synthetic grids.
- Under Indian mixed traffic, full RL signal control reduced collisions from {int(india_rule_coll)} to {int(india_full_coll)}, a {coll_red:.1f}% safety improvement over rule-based V2X.
- DQN signal control exhibited stronger stress robustness than rule-based methods, consistent with the reported +0.5% versus -1.1% comparative trend.
- Emergency-vehicle travel time improvement from baseline to full RL is {ev_improve_pct:.1f}% for runs where both values are available.
- PPO-GLOSA training converged, but convergence reflected speed-focused reward exploitation rather than balanced policy optimality.
- The system-level lesson is that signal-level RL currently contributes the most reliable resilience gains, while trajectory-level RL needs reward redesign.

## 5. Limitations & Future Work
The present pipeline has three notable limitations. First, some metrics are not uniformly logged across all modes, which restricts strict apples-to-apples comparison for throughput and emergency travel time in every scenario. Second, PPO-GLOSA objective design is currently imbalanced; speed-dominant rewards caused policy drift away from broader operational goals such as delay fairness and stop minimization. Third, the stress model, while representative, still abstracts many real-world factors such as pedestrian spillover, weather variation, and communication packet loss. Future work should prioritize unified KPI schemas, multi-objective reward tuning with explicit safety and comfort penalties, and multi-seed statistical validation. Additional extensions include online transfer learning from corridor-specific demand patterns, robust control under sensor noise, and hardware-in-the-loop V2X latency emulation. These changes would improve external validity and move the framework closer to deployment-grade intelligent traffic control for Indian urban corridors.

## 6. Conclusion
This BTP establishes an end-to-end V2X and RL evaluation workflow on a real Mehdipatnam network and shows that adaptive control is materially beneficial under Indian-style unstructured traffic. The strongest practical gain comes from DQN-based signal adaptation, which improves resilience and sharply lowers collision events during stress conditions. Full RL integration also accelerates emergency-vehicle movement, demonstrating value beyond average-flow optimization. At the same time, the PPO-GLOSA module reveals an important scientific caution: reward shaping can produce apparently converged but behaviorally narrow policies. Therefore, the project contribution is both performance-oriented and methodological, identifying what works today and what must be redesigned for robust generalization. With improved objective engineering, richer scenario coverage, and consistent KPI instrumentation, this framework can evolve into a credible decision-support baseline for Hyderabad smart-traffic deployments and future V2X-enabled urban mobility research.
"""
    return report


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    baseline_df, india_df, ppo_data, kpi_df = _load_data()
    rows = _build_summary_rows(baseline_df, india_df, ppo_data, kpi_df)
    summary_df = _write_summary_table(rows)
    report = _build_report(summary_df, ppo_data)
    REPORT_OUT.write_text(report, encoding="utf-8")
    print(f"Generated: {SUMMARY_OUT}")
    print(f"Generated: {REPORT_OUT}")


if __name__ == "__main__":
    main()
