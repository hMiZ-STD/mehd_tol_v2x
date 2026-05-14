# mehd_tol_v2x - AI-Enhanced V2X Simulation for Hyderabad's Mehdipatnam Corridor

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![SUMO](https://img.shields.io/badge/SUMO-1.22%2B-green)

## Overview
`mehd_tol_v2x` is a SUMO + Python/TraCI simulation project for testing V2X traffic intelligence on the Mehdipatnam corridor in Hyderabad. The current implementation is rule-based and serves as the baseline platform for upcoming AI/RL-driven control strategies.

## Features (Current Rule-Based)
- Adaptive traffic signals based on queue pressure.
- Green Light Optimal Speed Advisory (GLOSA) for non-EV vehicles.
- Emergency vehicle (EV) preemption and yielding logic.
- Corridor-level EV signal reservation management.
- V2N-style dynamic rerouting support.
- KPI logging for speed, waiting, congestion, EV activity, and throughput.

## Project Structure

| File/Path | Role |
|---|---|
| `main.py` | Simulation entrypoint; starts SUMO/TraCI and executes per-step control loop. |
| `config.py` | Centralized configuration and runtime feature toggles. |
| `adaptive_signals.py` | Rule-based adaptive traffic signal controller. |
| `glosa.py` | Rule-based GLOSA advisory logic. |
| `ev_preemption.py` | Emergency vehicle preemption and EV travel tracking. |
| `corridor_ev_manager.py` | Corridor-level EV coordination and signal protection. |
| `rerouter.py` | Dynamic rerouting logic for congestion response. |
| `kpi_logger.py` | KPI collection and CSV export to `outputs/`. |
| `network_graph.py` | Road graph utilities for trusted pool/path-level logic. |
| `generate_routes.py` | Route/trip generation helper script. |
| `mehd_tol.sumocfg` | SUMO scenario configuration entry file. |
| `mehd_tol.net.xml` | Generated SUMO road network for the corridor. |
| `mehd_tol.osm` | Source OSM map data. |
| `mehd_tol.rou.xml` | Vehicle route definitions. |
| `trips.trips.xml` | Trip definitions used by SUMO tools. |
| `vtypes.add.xml` | Vehicle type definitions (EV, bus, regular traffic, etc.). |
| `mehd_tol.view.xml` | SUMO GUI view settings. |
| `requirements.txt` | Python dependency manifest for project tooling/experiments. |
| `.gitignore` | Ignore rules for logs, outputs, cache, models, and local env files. |
| `outputs/` | Runtime logs and generated KPI files (created automatically). |

## Setup & Installation

### 1) Prerequisites
- Python 3.10 or newer.
- SUMO 1.22+ installed and available via `SUMO_HOME` or system `PATH`.

### 2) Clone and enter project
```bash
git clone https://github.com/hMiZ-STD/mehd_tol_v2x.git
cd mehd_tol_v2x
```

### 3) Create virtual environment (recommended)
```bash
python -m venv venv
```

Activate:
- Windows PowerShell:
```powershell
.\venv\Scripts\Activate.ps1
```
- Linux/macOS:
```bash
source venv/bin/activate
```

### 4) Install Python dependencies
```bash
pip install -r requirements.txt
```

### 5) Configure SUMO environment
Example:
- Windows PowerShell:
```powershell
$env:SUMO_HOME="C:\Program Files (x86)\Eclipse\Sumo"
```
- Linux/macOS:
```bash
export SUMO_HOME=/usr/share/sumo
```

## How to Run

### Run with GUI
```bash
python main.py
```

### Run without GUI
```bash
python main.py --no-gui
```

Outputs are written to:
- `outputs/sumo.log`
- `outputs/v2x_kpi_log.csv`

## KPI Metrics Explained
- `avg_speed_mps` / `avg_speed_kmh`: Mean speed across active vehicles.
- `halted_vehicles`: Vehicle count with near-zero speed.
- `congestion_pct`: Percentage of halted vehicles among active vehicles.
- `avg_wait_s`: Average waiting time over active vehicles.
- `active_evs`: Count of currently active emergency vehicles.
- `co2_mg_s`: Aggregate per-step CO2 emission signal from SUMO.
- `fuel_waste_index`: Proxy metric for idle fuel waste weighted by vehicle type.
- `total_departed_cum`: Cumulative number of departed vehicles.
- `total_arrived_cum`: Cumulative number of arrived vehicles.

## Roadmap

### Phase 0 - Cleanup
- Repository hygiene (`.gitignore`, output isolation, config cleanup).
- Standardized startup and robust TraCI error handling.
- Baseline documentation and dependency locking.

### Phase 1 - Baseline Benchmarking
- Freeze rule-based controller behavior as benchmark.
- Execute multiple demand scenarios.
- Produce repeatable KPI baselines for comparison.

### Phase 2 - DQN Adaptive Signal Control
- Introduce RL environment for intersection-level control.
- Train/evaluate DQN policies against Phase 1 benchmark.
- Add ablation and seed-based variance reporting.

### Phase 3 - PPO-GLOSA
- Add PPO-based longitudinal advisory policy.
- Joint evaluation with signal control strategies.
- Study travel-time/emissions trade-offs under mixed traffic.

### Phase 4 - KPI Dashboard
- Build Flask-based KPI dashboard and scenario explorer.
- Add run history, filtering, and visual comparison workflows.
- Export-ready summaries for experiments and reports.

## Contributing
- Fork and create a feature branch.
- Keep changes focused and add/adjust tests or reproducibility scripts where applicable.
- Use clear commit messages and open a PR with scenario, assumptions, and KPI impact.
- For major behavior changes, include before/after KPI snapshots.
