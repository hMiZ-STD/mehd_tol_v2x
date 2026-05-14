# ─── ALL PROJECT SETTINGS ─────────────────────────────────────────────────────
import os
import platform

# ─── SUMO BINARY AUTO-DETECTION ───────────────────────────────────────────────
def _find_sumo_binary(gui: bool = True) -> str:
    name = "sumo-gui" if gui else "sumo"
    ext  = ".exe" if platform.system() == "Windows" else ""

    sumo_home = os.environ.get("SUMO_HOME", "")
    if sumo_home:
        candidate = os.path.join(sumo_home, "bin", name + ext)
        if os.path.exists(candidate):
            return candidate

    win_paths = [
        r"C:\Program Files (x86)\Eclipse\Sumo",
        r"C:\Program Files\Eclipse\Sumo",
    ]                                               # ← FIXED: was missing ]
    linux_paths = ["/usr/share/sumo", "/opt/sumo", "/usr/local/share/sumo"]
    search = win_paths if platform.system() == "Windows" else linux_paths

    for base in search:
        candidate = os.path.join(base, "bin", name + ext)
        if os.path.exists(candidate):
            return candidate

    return name  # Fallback: assume SUMO is on system PATH

# ─── SIMULATION MODE ──────────────────────────────────────────────────────────
HEADLESS    = False                          # True = no GUI (sumo), False = sumo-gui
SUMO_BINARY = _find_sumo_binary(gui=not HEADLESS)
SUMO_CFG    = "mehd_tol.sumocfg"

# ─── SIMULATION TIMING ────────────────────────────────────────────────────────
SIM_STEPS   = 3600    # 1 hour of simulated traffic
STEP_LENGTH = 1.0     # Each step = 1 second
RANDOM_SEED = 42      # Reproducibility seed

# ─── GLOSA ────────────────────────────────────────────────────────────────────
GLOSA_MIN_SPEED = 5.0   # line ~30 — change from 2.0 to 5.0
GLOSA_LOOKAHEAD = 80.0  # line ~29 — change from 120.0 to 80.0
GLOSA_MAX_SPEED =  14.0   # m/s maximum advisory speed (~50 km/h)

# ─── EV PREEMPTION ────────────────────────────────────────────────────────────
EV_TYPE_ID        = "ev"    # must match vtypes_add.xml
EV_CLEAR_DISTANCE = 300.0   # metres ahead to force green corridor
EV_YIELD_DISTANCE =  30.0   # metres around EV where other vehicles must yield
EV_MAX_SPEED      =  25.0   # m/s cap on EV speed during preemption (~90 km/h)

# ─── INDIAN MIXED FLEET ───────────────────────────────────────────────────────
FLEET_2W_RATIO = 0.45   # Two-wheelers (motorcycle)
FLEET_EV_RATIO = 0.05   # Emergency vehicles
# Implicit car ratio = 1 - 0.45 - 0.05 = 0.50

# ─── KPI LOGGING ──────────────────────────────────────────────────────────────
KPI_LOG_INTERVAL = 30             # log one row every N simulation steps
OUTPUT_CSV       = "v2x_kpi_log.csv"

# ── Phase 1: Adaptive signal control ─────────────────────────────────────────
ADAPTIVE_SIGNALS_ENABLED = True
AS_MIN_GREEN            = 12.0   # s
AS_MAX_GREEN            = 45.0   # s
AS_EXTEND_STEP          = 6.0    # s
AS_ACTION_COOLDOWN      = 8.0    # s
AS_EMPTY_QUEUE          = 0      # halted vehicles on current green approaches
AS_SWITCH_QUEUE         = 4      # halted vehicles on competing approaches
AS_EXTEND_QUEUE         = 3      # halted vehicles to justify green extension

# ── Phase 1: EV corridor reservation layer ───────────────────────────────────
EV_CORRIDOR_ENABLED      = True
EV_CORRIDOR_LOOKAHEAD    = 600.0   # metres
EV_CORRIDOR_SIGNAL_LIMIT = 4       # reserve up to N upcoming signals
EV_CORRIDOR_ETA_LIMIT    = 60.0    # seconds
