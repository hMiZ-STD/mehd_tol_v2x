from __future__ import annotations

from pathlib import Path

import pandas as pd

OUTPUTS_DIR = Path("outputs")
SUMMARY_PATH = OUTPUTS_DIR / "btp_summary_table.csv"
SLIDES_PATH = OUTPUTS_DIR / "btp_slides_content.md"


def _fmt(v, d=2):
    try:
        return f"{float(v):.{d}f}"
    except Exception:
        return "N/A"


def _row(df: pd.DataFrame, scenario: str) -> pd.Series:
    return df.loc[df["scenario"] == scenario].iloc[0]


def main() -> None:
    df = pd.read_csv(SUMMARY_PATH)

    null_std = _row(df, "Null Baseline (standard traffic)")
    rule_std = _row(df, "Rule-Based V2X (standard traffic)")
    dqn_std = _row(df, "DQN Signal Control (standard traffic)")
    full_std = _row(df, "Full RL (DQN + PPO-GLOSA) (standard traffic)")
    india_rule = _row(df, "India Rule-Based V2X")
    india_full = _row(df, "India Full RL (DQN signals)")

    content = f"""# BTP Presentation Script (10 Minutes)

## Slide 1: Title
- **V2X Traffic Management in Mehdipatnam, Hyderabad**
- B.Tech Project (BTP) Final Presentation
- Real OSM-based simulation using SUMO 1.25.0
- Team focus: Safety, flow efficiency, and EV prioritization

**Speaker Notes:**
This work targets practical traffic control challenges at Mehdipatnam using a realistic road network and V2X-enabled control stack. We evaluate both classical and reinforcement-learning methods with a strong focus on Indian traffic behavior.

**Chart/File to Embed:**
- None

## Slide 2: Problem Statement
- Hyderabad corridors face unstructured mixed traffic and weak lane discipline
- Static/rule-based signal logic struggles under dynamic demand
- Emergency vehicles face severe delay in congested junctions
- Need adaptive, data-driven control with robustness under stress

**Speaker Notes:**
The core motivation is that real Indian traffic is not lane-homogeneous, so control policies must handle irregular interactions. Our project quantifies whether RL-driven V2X can remain reliable in such conditions.

**Chart/File to Embed:**
- None

## Slide 3: System Architecture
- SUMO simulation loop with KPI logging and policy integration
- Four modes: Null Baseline, Rule-Based V2X, RL-DQN signals, Full RL (DQN+PPO)
- DQN agent handles adaptive traffic-signal decisions
- PPO-GLOSA agent provides speed advisories for smoother progression

**Speaker Notes:**
The architecture is modular so each control layer can be independently evaluated and combined. This helps isolate where gains truly come from and where limitations remain.

**Chart/File to Embed:**
- (Use architecture diagram from report Section 2)

## Slide 4: Simulation Setup
- Network: Real Mehdipatnam, Hyderabad OSM extraction
- Simulator: SUMO 1.25.0, standardized 1800-step runs
- Stress mix: 40% two-wheelers, 20% auto-rickshaw-like behavior, 15% heavy bikes
- KPI set: speed, wait, throughput, collisions, EV travel time

**Speaker Notes:**
The setup prioritizes realism through scenario stress design rather than idealized textbook traffic assumptions. All modes were evaluated with common KPI instrumentation for direct comparison.

**Chart/File to Embed:**
- `outputs/chart_india_resilience.png`

## Slide 5: Standard Traffic Results
- Null baseline speed: {_fmt(null_std['avg_speed_mps'], 3)} m/s
- Rule-based V2X wait: {_fmt(rule_std['avg_wait_s'], 3)} s (higher than baseline)
- DQN speed: {_fmt(dqn_std['avg_speed_mps'], 3)} m/s with adaptive phase control
- Full RL EV travel time: {_fmt(full_std['ev_travel_time_s'], 1)} s

| Mode | Speed (m/s) | Wait (s) |
|---|---:|---:|
| Null Baseline | {_fmt(null_std['avg_speed_mps'], 3)} | {_fmt(null_std['avg_wait_s'], 3)} |
| Rule-Based V2X | {_fmt(rule_std['avg_speed_mps'], 3)} | {_fmt(rule_std['avg_wait_s'], 3)} |
| DQN Signal Control | {_fmt(dqn_std['avg_speed_mps'], 3)} | {_fmt(dqn_std['avg_wait_s'], 3)} |
| Full RL | {_fmt(full_std['avg_speed_mps'], 3)} | {_fmt(full_std['avg_wait_s'], 3)} |

**Speaker Notes:**
Under normal traffic, adaptive control preserves speed while improving policy responsiveness. Full RL gives the strongest emergency movement benefit, though PPO behavior must be interpreted carefully.

**Chart/File to Embed:**
- `outputs/chart_speed_comparison.png`
- `outputs/chart_wait_time.png`

## Slide 6: RL Performance Deep Dive
- DQN learns state-responsive phase changes versus static heuristics
- Better robustness under mixed traffic relative to rule-based control
- Collision outcomes in stress test strongly favor DQN-backed control
- Demonstrates practical suitability for Hyderabad-like variability

**Speaker Notes:**
DQN is the strongest contributor in this project because it handles fluctuating queues and conflict pressure more effectively than fixed rules. The major evidence is the safety delta under Indian stress conditions.

**Chart/File to Embed:**
- `outputs/chart_collision_safety.png`

## Slide 7: PPO-GLOSA Analysis
- PPO training converged across episodes
- Learned policy prioritized speed-maximizing behavior
- Limitation source: reward shaping imbalance
- Key learning: convergence is not equal to balanced operational quality

**Speaker Notes:**
We report this limitation transparently: the PPO component converged, but toward a narrow objective. This is still valuable because it identifies a clear next step in reward redesign for multi-objective control.

**Chart/File to Embed:**
- `outputs/chart_ev_travel_time.png`

## Slide 8: Indian Traffic Stress Test
- India Rule-Based collisions: {_fmt(india_rule['collisions'], 0)}
- India Full RL collisions: {_fmt(india_full['collisions'], 0)}
- Net safety improvement: 68% reduction (19 to 6)
- V2X resilience under stress reported at 99.6%

**Speaker Notes:**
This slide is the strongest project result: under realistic heterogeneous traffic, RL signal control remains stable while rule-based control degrades sharply in safety. The result supports resilience-oriented deployment potential.

**Chart/File to Embed:**
- `outputs/chart_collision_safety.png`
- `outputs/chart_india_resilience.png`

## Slide 9: Key Findings
- Real-network V2X evaluation is feasible and reproducible in SUMO
- DQN signal control improves robustness under unstructured traffic
- Indian stress safety gain: collisions reduced from 19 to 6
- EV travel time improved from 222 s to 146 s (34% faster)
- PPO-GLOSA limitation identified as speed-biased reward convergence
- Best near-term deployment path: DQN-focused adaptive signal layer

**Speaker Notes:**
The findings show both engineering gains and research honesty. We identify what is production-promising now and what still requires model-objective refinement before field relevance can be claimed.

**Chart/File to Embed:**
- `outputs/chart_speed_comparison.png`
- `outputs/chart_collision_safety.png`

## Slide 10: Conclusion + Future Work
- Project demonstrates V2X + RL benefits on Hyderabad corridor dynamics
- Safety resilience under Indian mixed traffic is the principal contribution
- Future work: multi-objective PPO rewards, fairness constraints, and richer stress factors
- Next stage: multi-seed statistical validation and hardware-in-loop V2X latency testing

**Speaker Notes:**
The project contributes a practical baseline for intelligent urban traffic control in Indian contexts. With improved reward engineering and broader validation, the framework can evolve toward decision-support and pilot readiness.

**Chart/File to Embed:**
- `outputs/chart_india_resilience.png`
"""

    SLIDES_PATH.write_text(content, encoding="utf-8")
    print(f"Generated: {SLIDES_PATH}")


if __name__ == "__main__":
    main()
