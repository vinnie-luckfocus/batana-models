#!/usr/bin/env python3
"""End-to-end validation for PRD 01."""

import json
import sys
from pathlib import Path

def validate() -> int:
    logs_dir = Path("logs")
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Load QC report
    qc_path = logs_dir / "qc_report.json"
    if not qc_path.exists():
        print("ERROR: QC report not found")
        return 1

    with open(qc_path) as f:
        qc = json.load(f)

    total = len(qc)
    usable = sum(1 for v in qc.values() if v.get("status") != "UNREFINABLE")
    print(f"Total videos: {total}, Usable: {usable}")

    if usable < 70:
        print(f"WARNING: Usable videos {usable} < 70")

    # Check latency profile
    latency_path = logs_dir / "latency_profile.txt"
    latency_ok = latency_path.exists()
    if latency_ok:
        print("Latency profile found")
    else:
        print("WARNING: Latency profile not found")

    # Check cleaning log
    cleaning_log = logs_dir / "2d_cleaning_log.csv"
    cleaning_ok = cleaning_log.exists() and cleaning_log.stat().st_size > 0
    if cleaning_ok:
        print("2D cleaning log found")
    else:
        print("WARNING: 2D cleaning log not found")

    # Check visualizations
    viz_train = list(Path("visuals/2d/train").glob("*.mp4"))
    viz_val = list(Path("visuals/2d/val").glob("*.mp4"))
    print(f"Visualization videos: train={len(viz_train)}, val={len(viz_val)}")

    report = {
        "total_videos": total,
        "usable_videos": usable,
        "latency_profile_found": latency_ok,
        "cleaning_log_found": cleaning_ok,
        "viz_train_count": len(viz_train),
        "viz_val_count": len(viz_val),
    }

    report_path = reports_dir / "2d_phase1_validation.md"
    with open(report_path, "w") as f:
        f.write("# PRD 01 Phase 1 Validation Report\n\n")
        f.write(f"- Total videos: {total}\n")
        f.write(f"- Usable videos: {usable}\n")
        f.write(f"- Latency profile: {'OK' if latency_ok else 'MISSING'}\n")
        f.write(f"- Cleaning log: {'OK' if cleaning_ok else 'MISSING'}\n")
        f.write(f"- Visualizations train: {len(viz_train)}\n")
        f.write(f"- Visualizations val: {len(viz_val)}\n")

    print(f"Report written to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(validate())
