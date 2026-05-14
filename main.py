"""
main.py — V2X Simulation Entry Point
"""

import os
import sys
import shutil
import traci

from config import SUMO_BINARY, SUMO_CFG, SIM_STEPS, KPI_LOG_INTERVAL
from network_graph import build_trusted_pool
from glosa import apply_glosa
from ev_preemption import apply_ev_preemption, get_ev_summary
from kpi_logger import log_step, save_csv, kpi_log
from rerouter import apply_rerouting
from adaptive_signals import apply_adaptive_signals
from corridor_ev_manager import update_ev_corridors


def _check_binary() -> None:
    if os.path.exists(SUMO_BINARY):
        return
    if shutil.which(SUMO_BINARY):
        return
    print(f"ERROR: SUMO binary not found: '{SUMO_BINARY}'")
    sys.exit(1)


def start_sumo() -> None:
    _check_binary()
    traci.start([
        SUMO_BINARY, "-c", SUMO_CFG,
        "--collision.action", "warn",
        "--no-step-log", "false",
        "--no-warnings",
        "--message-log", "sumo.log",
    ])
    print("✅ SUMO launched. (Warnings → sumo.log)")


def run() -> None:
    start_sumo()
    build_trusted_pool()

    print(f"\n🚦 V2X Simulation — {SIM_STEPS} steps\n")

    step = 0
    try:
        for step in range(SIM_STEPS):
            traci.simulationStep()

            # Phase 1-A: reserve TLS ahead of EVs so adaptive control does not interfere
            protected_tls = update_ev_corridors()

            # Phase 1-B: adaptive signal control for normal traffic
            apply_adaptive_signals(protected_tls)

            # Existing V2X logic stays in place
            vehicles = traci.vehicle.getIDList()
            for veh_id in vehicles:
                try:
                    if traci.vehicle.getTypeID(veh_id) != "ev":
                        apply_glosa(veh_id)
                except traci.TraCIException:
                    pass

            active_evs = apply_ev_preemption()
            apply_rerouting()

            if step % KPI_LOG_INTERVAL == 0:
                log_step(step, active_evs)
                if kpi_log:
                    row = kpi_log[-1]
                    print(
                        f"  Step {row['step']:4d} | "
                        f"Vehs: {row['total_vehicles']:3d} | "
                        f"Avg: {row['avg_speed_kmh']:5.1f} km/h | "
                        f"Halted: {row['halted_vehicles']:3d} | "
                        f"Wait: {row['avg_wait_s']:5.1f} s | "
                        f"EVs: {row['active_evs']}"
                    )

    except traci.TraCIException as e:
        print(f"\n⚠️  TraCI error at step {step}: {e}")

    finally:
        try:
            traci.close()
        except Exception:
            pass

    print("\n" + "=" * 60)
    print("  ✅ SIMULATION COMPLETE")
    print("=" * 60)
    print(f"  Avg EV Travel Time : {get_ev_summary():.1f} s")
    save_csv()
    print("=" * 60)


if __name__ == "__main__":
    run()
