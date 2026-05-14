"""V2X simulation entrypoint for SUMO + TraCI."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import traci

import config
from adaptive_signals import apply_adaptive_signals
from corridor_ev_manager import update_ev_corridors
from ev_preemption import apply_ev_preemption, get_ev_summary
from glosa import apply_glosa
from kpi_logger import kpi_log, log_step, save_csv
from network_graph import build_trusted_pool
from rerouter import apply_rerouting


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mehd_tol_v2x SUMO simulation.")
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run headless with `sumo` instead of `sumo-gui`.",
    )
    return parser.parse_args()


def _resolve_sumo_binary(no_gui: bool) -> str:
    if no_gui:
        return config.SUMO_BINARY_NO_GUI
    return config.SUMO_BINARY_GUI


def _check_binary(binary: str) -> None:
    if os.path.exists(binary):
        return
    if shutil.which(binary):
        return
    raise FileNotFoundError(
        f"SUMO binary not found: '{binary}'. Set SUMO_HOME or add SUMO to PATH."
    )


def _check_cfg_exists() -> None:
    cfg_path = Path(config.SUMO_CFG)
    if not cfg_path.exists():
        raise FileNotFoundError(f"SUMO config file not found: '{cfg_path}'")


def _ensure_output_dir() -> None:
    Path(config.LOG_DIR).mkdir(parents=True, exist_ok=True)


def start_sumo(no_gui: bool) -> None:
    _ensure_output_dir()
    _check_cfg_exists()

    sumo_binary = _resolve_sumo_binary(no_gui=no_gui)
    _check_binary(sumo_binary)

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
        config.SUMO_MESSAGE_LOG,
    ]

    try:
        traci.start(cmd)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to start TraCI/SUMO with command: {' '.join(cmd)}"
        ) from exc

    mode = "headless (sumo)" if no_gui else "GUI (sumo-gui)"
    print(f"SUMO launched in {mode} mode. Warnings -> {config.SUMO_MESSAGE_LOG}")


def run(no_gui: bool = False) -> None:
    step = 0
    try:
        start_sumo(no_gui=no_gui)
        build_trusted_pool()

        print(f"\nV2X Simulation - {config.SIM_STEPS} steps\n")

        for step in range(config.SIM_STEPS):
            traci.simulationStep()

            protected_tls = update_ev_corridors()
            apply_adaptive_signals(protected_tls)

            for veh_id in traci.vehicle.getIDList():
                try:
                    if traci.vehicle.getTypeID(veh_id) != config.EV_TYPE_ID:
                        apply_glosa(veh_id)
                except traci.TraCIException:
                    continue

            active_evs = apply_ev_preemption()
            apply_rerouting()

            if step % config.KPI_LOG_INTERVAL == 0:
                log_step(step, active_evs)
                if kpi_log:
                    row = kpi_log[-1]
                    print(
                        f"Step {row['step']:4d} | "
                        f"Vehs: {row['total_vehicles']:3d} | "
                        f"Avg: {row['avg_speed_kmh']:5.1f} km/h | "
                        f"Halted: {row['halted_vehicles']:3d} | "
                        f"Wait: {row['avg_wait_s']:5.1f} s | "
                        f"EVs: {row['active_evs']}"
                    )

    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\nInterrupt received at step {step}. Shutting down gracefully...")
    except traci.TraCIException as exc:
        print(f"TraCI error at step {step}: {exc}")
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
    run(no_gui=args.no_gui)
