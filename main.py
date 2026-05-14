"""V2X simulation entrypoint for SUMO + TraCI."""

from __future__ import annotations

import argparse
import os
import sys

import sumolib
import traci

import config
from adaptive_signals import apply_adaptive_signals
from corridor_ev_manager import update_ev_corridors
from ev_preemption import apply_ev_preemption, get_ev_summary
from glosa import apply_glosa
from kpi_logger import kpi_log, log_step, reset_kpi_log, save_csv, set_output_csv
from network_graph import build_trusted_pool
from rerouter import apply_rerouting

VALID_MODES = ("null_baseline", "rule_based_v2x", "rule_based_no_ev")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mehd_tol_v2x SUMO simulation.")
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run headless with `sumo` instead of `sumo-gui`.",
    )
    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        default="rule_based_v2x",
        help="Simulation mode preset for baseline benchmarking.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=config.SIM_STEPS,
        help="Number of simulation steps to run.",
    )
    return parser.parse_args()


def _check_cfg_exists() -> None:
    if not os.path.exists(config.SUMO_CFG):
        raise FileNotFoundError(f"SUMO config file not found: '{config.SUMO_CFG}'")


def _ensure_output_dir() -> None:
    os.makedirs(config.LOG_DIR, exist_ok=True)


def _apply_mode_flags(mode: str) -> dict:
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Expected one of: {VALID_MODES}")

    # Keep config flags aligned for traceability across scripts.
    config.BASELINE_MODE = mode != "rule_based_v2x"
    config.USE_RL_SIGNALS = False
    config.USE_RL_GLOSA = False

    if mode == "null_baseline":
        return {
            "adaptive_signals": False,
            "glosa": False,
            "ev_preemption": False,
            "rerouting": False,
            "ev_corridor": False,
        }
    if mode == "rule_based_no_ev":
        return {
            "adaptive_signals": True,
            "glosa": True,
            "ev_preemption": False,
            "rerouting": False,
            "ev_corridor": False,
        }
    return {
        "adaptive_signals": True,
        "glosa": True,
        "ev_preemption": True,
        "rerouting": True,
        "ev_corridor": True,
    }


def start_sumo(no_gui: bool) -> None:
    _ensure_output_dir()
    _check_cfg_exists()

    binary_name = "sumo" if no_gui else "sumo-gui"
    try:
        sumo_binary = sumolib.checkBinary(binary_name)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to resolve SUMO binary '{binary_name}'. "
            "Ensure SUMO is installed and SUMO_HOME/PATH is set."
        ) from exc

    cmd = [
        sumo_binary,
        "-c",
        config.SUMO_CFG,
        "--collision.action",
        "warn",
        "--no-step-log",
        "false",
        "--no-warnings",
        "--message-log",
        os.path.join(config.LOG_DIR, "sumo.log"),
    ]

    try:
        traci.start(cmd)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to start TraCI/SUMO with command: {' '.join(cmd)}"
        ) from exc

    run_mode = "headless (sumo)" if no_gui else "GUI (sumo-gui)"
    print(f"SUMO launched in {run_mode} mode.")


def run(
    no_gui: bool = False,
    mode: str = "rule_based_v2x",
    sim_steps: int | None = None,
    output_csv: str | None = None,
    progress_interval: int = 300,
) -> None:
    steps = sim_steps if sim_steps is not None else config.SIM_STEPS
    features = _apply_mode_flags(mode)
    step = 0
    reset_kpi_log()
    if output_csv:
        set_output_csv(output_csv)

    try:
        start_sumo(no_gui=no_gui)
        build_trusted_pool()
        print(f"\nV2X Simulation - mode={mode}, steps={steps}\n")

        for step in range(steps):
            traci.simulationStep()
            protected_tls = set()

            if features["ev_corridor"]:
                protected_tls = update_ev_corridors()

            if features["adaptive_signals"]:
                apply_adaptive_signals(protected_tls)

            if features["glosa"]:
                for veh_id in traci.vehicle.getIDList():
                    try:
                        if traci.vehicle.getTypeID(veh_id) != config.EV_TYPE_ID:
                            apply_glosa(veh_id)
                    except traci.TraCIException:
                        continue

            active_evs = []
            if features["ev_preemption"]:
                active_evs = apply_ev_preemption()

            if features["rerouting"]:
                apply_rerouting()

            if step % config.KPI_LOG_INTERVAL == 0:
                log_step(step, active_evs)

            if step > 0 and step % progress_interval == 0:
                print(f"[{mode}] progress: step {step}/{steps}")

    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    except traci.TraCIException as exc:
        print(f"TraCI error at step {step}: {exc}")
    except KeyboardInterrupt:
        print(f"\nInterrupt received at step {step}. Shutting down gracefully...")
    finally:
        try:
            traci.close()
        except Exception:
            pass

    print("\n" + "=" * 60)
    print("  SIMULATION COMPLETE")
    print("=" * 60)
    print(f"  Avg EV Travel Time : {get_ev_summary():.1f} s")
    save_csv()
    print("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    run(no_gui=args.no_gui, mode=args.mode, sim_steps=args.steps)
