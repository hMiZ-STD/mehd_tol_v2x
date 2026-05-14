"""Per-step KPI collection and CSV export."""

from __future__ import annotations

import csv
from pathlib import Path

import traci

from config import EV_TYPE_ID, LOG_DIR, OUTPUT_CSV

kpi_log: list = []
_output_csv_path = OUTPUT_CSV

_FUEL_FACTOR = {"2W": 0.30, "car": 0.50, EV_TYPE_ID: 0.10}
_DEFAULT_FUEL = 0.50

_cum_departed = 0
_cum_arrived = 0


def set_output_csv(path: str) -> None:
    global _output_csv_path
    _output_csv_path = path


def reset_kpi_log() -> None:
    global _cum_departed, _cum_arrived
    kpi_log.clear()
    _cum_departed = 0
    _cum_arrived = 0


def log_step(step: int, active_evs: list) -> None:
    global _cum_departed, _cum_arrived

    all_vehicles = traci.vehicle.getIDList()
    n = len(all_vehicles)
    if n == 0:
        return

    speeds = [traci.vehicle.getSpeed(v) for v in all_vehicles]
    wait_times = [traci.vehicle.getWaitingTime(v) for v in all_vehicles]

    avg_speed = sum(speeds) / n
    halted = sum(1 for speed in speeds if speed < 0.1)
    avg_wait = sum(wait_times) / n
    congestion = round(halted / n * 100, 2)

    co2_total = -1.0
    try:
        co2_total = round(sum(traci.vehicle.getCO2Emission(v) for v in all_vehicles), 2)
    except Exception:
        pass

    fuel_waste = 0.0
    for vehicle_id in all_vehicles:
        if traci.vehicle.getSpeed(vehicle_id) < 0.1:
            try:
                vtype = traci.vehicle.getTypeID(vehicle_id)
            except traci.TraCIException:
                vtype = "car"
            fuel_waste += _FUEL_FACTOR.get(vtype, _DEFAULT_FUEL)

    _cum_departed += traci.simulation.getDepartedNumber()
    _cum_arrived += traci.simulation.getArrivedNumber()

    kpi_log.append(
        {
            "step": step,
            "sim_time_s": traci.simulation.getTime(),
            "total_vehicles": n,
            "avg_speed_mps": round(avg_speed, 3),
            "avg_speed_kmh": round(avg_speed * 3.6, 3),
            "halted_vehicles": halted,
            "congestion_pct": congestion,
            "avg_wait_s": round(avg_wait, 2),
            "active_evs": len(active_evs),
            "co2_mg_s": co2_total,
            "fuel_waste_index": round(fuel_waste, 2),
            "total_departed_cum": _cum_departed,
            "total_arrived_cum": _cum_arrived,
        }
    )


def save_csv() -> None:
    if not kpi_log:
        print("No KPI data to save.")
        return

    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    output_path = Path(_output_csv_path)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=kpi_log[0].keys())
        writer.writeheader()
        writer.writerows(kpi_log)

    print(f"KPI data saved -> {output_path} ({len(kpi_log)} rows)")
