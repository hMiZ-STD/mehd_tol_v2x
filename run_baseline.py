"""Run baseline benchmark modes for mehd_tol_v2x."""

from __future__ import annotations

import os
import traceback

import config
from main import run

MODES = ("null_baseline", "rule_based_v2x", "rule_based_no_ev")
RUN_STEPS = config.SIM_STEPS
PROGRESS_INTERVAL = 300


def _output_csv_for_mode(mode: str) -> str:
    return os.path.join(config.LOG_DIR, f"{mode}_kpi.csv")


def main() -> None:
    os.makedirs(config.LOG_DIR, exist_ok=True)

    for mode in MODES:
        output_csv = _output_csv_for_mode(mode)
        print("\n" + "#" * 70)
        print(f"Starting baseline mode: {mode}")
        print(f"Output CSV: {output_csv}")
        print("#" * 70)
        try:
            run(
                no_gui=True,
                mode=mode,
                sim_steps=RUN_STEPS,
                output_csv=output_csv,
                progress_interval=PROGRESS_INTERVAL,
            )
        except Exception as exc:
            print(f"[ERROR] Mode '{mode}' failed due to SUMO/TraCI issue: {exc}")
            traceback.print_exc()
            continue

    print("\nBaseline run complete.")


if __name__ == "__main__":
    main()
