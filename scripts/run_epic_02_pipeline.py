#!/usr/bin/env python3
"""Orchestrator for Epic 02: dataset prep, training, validation, reporting, and self-correction."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_TC_TARGET = 0.82
_KS_TARGET = 0.92
_MR_TARGET = 0.05


def _log(logger_path: Path, message: str) -> None:
    """Append a timestamped message to the pipeline log."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    logger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(logger_path, "a", encoding="utf-8") as f:
        f.write(f"{now} {message}\n")


def _run_command(cmd: list[str], log_path: Path) -> int:
    """Run a subprocess command and log the outcome."""
    _log(log_path, f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        _log(log_path, f"[FAIL] rc={result.returncode} cmd={' '.join(cmd)}")
    else:
        _log(log_path, f"[OK] cmd={' '.join(cmd)}")
    return result.returncode


def _validation_failed(metrics_path: Path) -> bool:
    """Return True if validation metrics miss any target."""
    if not metrics_path.is_file():
        return True
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return True
    agg = data.get("aggregate", {})
    tc = agg.get("temporal_coherence", 0.0)
    ks = agg.get("keypoint_stability", 0.0)
    mr = agg.get("missing_rate", 1.0)
    return tc <= _TC_TARGET or ks <= _KS_TARGET or mr >= _MR_TARGET


def _run_phase01(data_root: Path, log_path: Path) -> int:
    """Run Phase 01 anomaly detection and strict correction."""
    _log(log_path, "SELF_CORRECTION running Phase 01 anomaly detection")
    rc = _run_command(
        [sys.executable, "scripts/detect_anomalies_2d.py", "--data-root", str(data_root)],
        log_path,
    )
    if rc != 0:
        _log(log_path, "SELF_CORRECTION_EXHAUSTED anomaly detection failed")
        return rc

    _log(log_path, "SELF_CORRECTION running Phase 01 trajectory correction --strict-mode")
    rc = _run_command(
        [
            sys.executable,
            "scripts/correct_trajectories_2d.py",
            "--data-root",
            str(data_root),
            "--strict-mode",
        ],
        log_path,
    )
    if rc != 0:
        _log(log_path, "SELF_CORRECTION_EXHAUSTED trajectory correction failed")
    return rc


def _run_dataset_prep(data_root: Path, log_path: Path) -> int:
    return _run_command(
        [sys.executable, "scripts/prepare_mmpose_dataset.py", "--data-root", str(data_root)],
        log_path,
    )


def _run_training(log_path: Path) -> int:
    return _run_command(
        [
            sys.executable,
            "scripts/train_2d_pose.py",
            "--config",
            "configs/rtmpose_m_finetune_baseball.py",
        ],
        log_path,
    )


def _run_validation(data_root: Path, metrics_path: Path, log_path: Path) -> int:
    return _run_command(
        [
            sys.executable,
            "scripts/validate_2d_pose.py",
            "--model",
            "models/2d_pose/rtmpose_m_finetuned.pth",
            "--val-dir",
            str(data_root / "mmpose_baseball" / "images" / "val"),
            "--output",
            str(metrics_path),
        ],
        log_path,
    )


def _run_report(metrics_path: Path, report_path: Path, log_path: Path) -> int:
    return _run_command(
        [
            sys.executable,
            "scripts/generate_validation_report.py",
            "--metrics-json",
            str(metrics_path),
            "--output",
            str(report_path),
        ],
        log_path,
    )


def run_pipeline(
    data_root: Path,
    log_path: Path,
    metrics_path: Path,
    report_path: Path,
    skip_training: bool = False,
    skip_validation: bool = False,
) -> int:
    """Execute the full Epic 02 pipeline with optional self-correction."""
    _log(log_path, "PIPELINE_START")

    # Step 1: Dataset preparation
    rc = _run_dataset_prep(data_root, log_path)
    if rc != 0:
        _log(log_path, "PIPELINE_HALTED dataset_prep failed")
        return 1

    # Step 2: Training
    if not skip_training:
        rc = _run_training(log_path)
        if rc != 0:
            _log(log_path, "PIPELINE_HALTED training failed")
            return 1

    # Step 3: Validation
    if not skip_validation:
        rc = _run_validation(data_root, metrics_path, log_path)
        if rc != 0:
            _log(log_path, "VALIDATION_FAILED metric targets not met")
        else:
            _log(log_path, "VALIDATION_PASSED")

    # Step 4: Report
    if metrics_path.is_file():
        _run_report(metrics_path, report_path, log_path)

    # Self-correction loop
    if _validation_failed(metrics_path):
        _log(log_path, "VALIDATION_FAILED triggering self-correction")

        rc = _run_phase01(data_root, log_path)
        if rc != 0:
            return 1

        # Re-run dataset prep
        _log(log_path, "SELF_CORRECTION re-running dataset prep")
        rc = _run_dataset_prep(data_root, log_path)
        if rc != 0:
            _log(log_path, "SELF_CORRECTION_EXHAUSTED dataset prep retry failed")
            return 1

        # Re-run training
        if not skip_training:
            _log(log_path, "SELF_CORRECTION re-running training")
            rc = _run_training(log_path)
            if rc != 0:
                _log(log_path, "SELF_CORRECTION_EXHAUSTED training retry failed")
                return 1

        # Re-run validation (always validate after self-correction to verify fix)
        _log(log_path, "SELF_CORRECTION re-running validation")
        rc = _run_validation(data_root, metrics_path, log_path)
        if rc != 0:
            _log(log_path, "VALIDATION_FAILED_RETRY")

        # Re-generate report
        if metrics_path.is_file():
            _run_report(metrics_path, report_path, log_path)

        if _validation_failed(metrics_path):
            _log(log_path, "SELF_CORRECTION_EXHAUSTED validation still failing after retry")
            return 1
        _log(log_path, "SELF_CORRECTION_SUCCEEDED")

    _log(log_path, "PIPELINE_COMPLETE")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Epic 02 end-to-end pipeline")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args(argv)

    log_path = Path("logs/pipeline.log")
    metrics_path = Path("logs/validation_metrics.json")
    report_path = Path("reports/2d_validation_report.md")

    return run_pipeline(
        data_root=args.data_root,
        log_path=log_path,
        metrics_path=metrics_path,
        report_path=report_path,
        skip_training=args.skip_training,
        skip_validation=args.skip_validation,
    )


if __name__ == "__main__":
    sys.exit(main())
