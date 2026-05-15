from __future__ import annotations

import subprocess
import sys
from pathlib import Path

OUTPUTS_DIR = Path("outputs")

EXPECTED = [
    OUTPUTS_DIR / "btp_summary_table.csv",
    OUTPUTS_DIR / "btp_final_report.md",
    OUTPUTS_DIR / "chart_speed_comparison.png",
    OUTPUTS_DIR / "chart_wait_time.png",
    OUTPUTS_DIR / "chart_collision_safety.png",
    OUTPUTS_DIR / "chart_ev_travel_time.png",
    OUTPUTS_DIR / "chart_india_resilience.png",
    OUTPUTS_DIR / "btp_slides_content.md",
]


def _run(cmd: list[str]) -> None:
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _size_str(path: Path) -> str:
    b = path.stat().st_size
    if b < 1024:
        return f"{b} B"
    if b < 1024 * 1024:
        return f"{b / 1024:.1f} KB"
    return f"{b / (1024 * 1024):.2f} MB"


def main() -> None:
    _run([sys.executable, "generate_report.py"])
    _run([sys.executable, "generate_charts.py"])
    _run([sys.executable, "generate_slides_content.py"])

    print("\n=== Phase 5 Complete ===")
    for path in EXPECTED:
        if path.exists():
            print(f"[x] {path.as_posix()} ({_size_str(path)})")
        else:
            print(f"[ ] {path.as_posix()} (0 B)")
            print(f"WARNING: {path.as_posix()} not generated")


if __name__ == "__main__":
    main()
