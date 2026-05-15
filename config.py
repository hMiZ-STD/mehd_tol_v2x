"""Central project configuration for mehd_tol_v2x."""

from __future__ import annotations

import os
import platform
from pathlib import Path


def _find_sumo_binary(gui: bool = True) -> str:
    """Return SUMO binary path if found, otherwise fallback to executable name."""
    name = "sumo-gui" if gui else "sumo"
    ext = ".exe" if platform.system() == "Windows" else ""

    sumo_home = os.environ.get("SUMO_HOME", "")
    if sumo_home:
        candidate = Path(sumo_home) / "bin" / f"{name}{ext}"
        if candidate.exists():
            return str(candidate)

    win_paths = [
        r"C:\Program Files (x86)\Eclipse\Sumo",
        r"C:\Program Files\Eclipse\Sumo",
    ]
    linux_paths = ["/usr/share/sumo", "/opt/sumo", "/usr/local/share/sumo"]
    search = win_paths if platform.system() == "Windows" else linux_paths

    for base in search:
        candidate = Path(base) / "bin" / f"{name}{ext}"
        if candidate.exists():
            return str(candidate)

    return name


# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
SUMO_CFG = str(PROJECT_ROOT / "mehd_tol.sumocfg")
LOG_DIR = "outputs/"
OUTPUT_CSV = str(Path(LOG_DIR) / "v2x_kpi_log.csv")
SUMO_MESSAGE_LOG = str(Path(LOG_DIR) / "sumo.log")
MODEL_DIR = "outputs/models"

# Runtime mode toggles
HEADLESS = False
BASELINE_MODE = False
USE_RL_SIGNALS = False
USE_RL_GLOSA = False
MIN_GREEN_STEPS = 10

# SUMO binary defaults
SUMO_BINARY_GUI = _find_sumo_binary(gui=True)
SUMO_BINARY_NO_GUI = _find_sumo_binary(gui=False)
SUMO_BINARY = SUMO_BINARY_NO_GUI if HEADLESS else SUMO_BINARY_GUI

# Simulation timing
SIM_STEPS = 3600
STEP_LENGTH = 1.0
RANDOM_SEED = 42

# GLOSA
GLOSA_MIN_SPEED = 5.0
GLOSA_LOOKAHEAD = 80.0
GLOSA_MAX_SPEED = 14.0

# EV preemption
EV_TYPE_ID = "ev"
EV_CLEAR_DISTANCE = 300.0
EV_YIELD_DISTANCE = 30.0
EV_MAX_SPEED = 25.0

# Indian mixed fleet mix
FLEET_2W_RATIO = 0.45
FLEET_EV_RATIO = 0.05

# KPI logging
KPI_LOG_INTERVAL = 30

# Phase 1: Adaptive signal control
ADAPTIVE_SIGNALS_ENABLED = True
AS_MIN_GREEN = 12.0
AS_MAX_GREEN = 45.0
AS_EXTEND_STEP = 6.0
AS_ACTION_COOLDOWN = 8.0
AS_EMPTY_QUEUE = 0
AS_SWITCH_QUEUE = 4
AS_EXTEND_QUEUE = 3

# Phase 1: EV corridor reservation layer
EV_CORRIDOR_ENABLED = True
EV_CORRIDOR_LOOKAHEAD = 600.0
EV_CORRIDOR_SIGNAL_LIMIT = 4
EV_CORRIDOR_ETA_LIMIT = 60.0
