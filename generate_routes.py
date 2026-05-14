# why this file?
# This creates random route for each vehicle and changes destination for every turn
# why randomness needed?
# I mean why not ? , wouldnt it be better , this file to run once before running main.py , not necessarily everytime
# It uses existing randomTrips.py to generate random trips

import os
import subprocess
import sys
import random
import re

from config import RANDOM_SEED, FLEET_2W_RATIO, FLEET_EV_RATIO

# ─── LOCATE SUMO ──────────────────────────────────────────────────────────────
sumo_home = os.environ.get("SUMO_HOME", "")
if not sumo_home:
    search_paths = [
        r"C:\Program Files (x86)\Eclipse\Sumo",
        r"C:\Program Files\Eclipse\Sumo",
        "/usr/share/sumo",
        "/opt/sumo",
        "/usr/local/share/sumo",
    ]   

    for path in search_paths:
        if os.path.exists(path):
            sumo_home = path
            break

if not sumo_home:
    print("ERROR: Cannot find SUMO. Set the SUMO_HOME environment variable.")
    sys.exit(1)

random_trips_script = os.path.join(sumo_home, "tools", "randomTrips.py")
if not os.path.exists(random_trips_script):
    print(f"ERROR: randomTrips.py not found at: {random_trips_script}")
    sys.exit(1)

print(f"✅ Found SUMO at: {sumo_home}")

# ─── REMOVE OLD OUTPUT FILES ──────────────────────────────────────────────────
for f in ["mehd_tol.rou.xml", "trips.trips.xml"]:
    if os.path.exists(f):
        os.remove(f)
        print(f"🗑️  Deleted old {f}")

# ─── GENERATE VALIDATED TRIPS ─────────────────────────────────────────────────
print("Generating 250 validated vehicles spread across 1 hour...")
cmd = [
    sys.executable, random_trips_script,
    "-n", "mehd_tol.net.xml",
    "-b", "0",
    "-e", "3600",
    "-p", "14.4",
    "-r", "mehd_tol.rou.xml",
    "-o", "trips.trips.xml",
    "--seed",         str(RANDOM_SEED),
    "--validate",
    "--min-distance", "200",
    "--random-depart",
]                                               # ← FIXED: was missing ]

result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode != 0:
    print("ERROR from randomTrips.py:")
    print(result.stderr[-2000:])
    sys.exit(1)

print("✅ Routes generated and validated.")

# ─── INJECT INDIAN MIXED FLEET ────────────────────────────────────────────────
car_ratio = 1.0 - FLEET_2W_RATIO - FLEET_EV_RATIO
assert car_ratio > 0, "Fleet ratios must sum to < 1.0"

print(
    f"Assigning fleet: "
    f"{int(FLEET_2W_RATIO*100)}% 2W | "
    f"{int(car_ratio*100)}% car | "
    f"{int(FLEET_EV_RATIO*100)}% EV ..."
)                                               # ← FIXED: was missing )

random.seed(RANDOM_SEED)

with open("mehd_tol.rou.xml", "r", encoding="utf-8") as f:
    content = f.read()

lines      = content.split("\n")
new_lines  = []
total      = 0
type_counts = {"2W": 0, "car": 0, "ev": 0}

_2W_thresh = FLEET_2W_RATIO
_EV_thresh = FLEET_2W_RATIO + FLEET_EV_RATIO

for line in lines:
    if '<vehicle ' in line and 'type=' not in line:
        r = random.random()
        if r < _2W_thresh:
            vtype = "2W"
        elif r < _EV_thresh:
            vtype = "ev"
        else:
            vtype = "car"
        line = re.sub(r'(<vehicle\s)', rf'\1type="{vtype}" ', line)
        type_counts[vtype] += 1
        total += 1
    new_lines.append(line)

with open("mehd_tol.rou.xml", "w", encoding="utf-8") as f:
    f.write("\n".join(new_lines))

print(f"✅ Fleet assigned to {total} vehicles:")
print(f"   🏍️  2W  : {type_counts['2W']}")
print(f"   🚗 car  : {type_counts['car']}")
print(f"   🚑 EV   : {type_counts['ev']}")
print("Done. Run: python main.py")
