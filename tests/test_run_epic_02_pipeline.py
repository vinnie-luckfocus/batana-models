import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_epic_02_pipeline import (
    _log,
    _validation_failed,
    main,
    run_pipeline,
)


class TestValidationFailed:
    def test_missing_file_is_failed(self, tmp_path: Path) -> None:
        assert _validation_failed(tmp_path / "missing.json") is True

    def test_passing_metrics(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.json"
        path.write_text(
            json.dumps(
                {
                    "aggregate": {
                        "temporal_coherence": 0.85,
                        "keypoint_stability": 0.93,
                        "missing_rate": 0.04,
                    }
                }
            ),
            encoding="utf-8",
        )
        assert _validation_failed(path) is False

    def test_tc_fail(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.json"
        path.write_text(
            json.dumps(
                {
                    "aggregate": {
                        "temporal_coherence": 0.80,
                        "keypoint_stability": 0.93,
                        "missing_rate": 0.04,
                    }
                }
            ),
            encoding="utf-8",
        )
        assert _validation_failed(path) is True

    def test_ks_fail(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.json"
        path.write_text(
            json.dumps(
                {
                    "aggregate": {
                        "temporal_coherence": 0.85,
                        "keypoint_stability": 0.90,
                        "missing_rate": 0.04,
                    }
                }
            ),
            encoding="utf-8",
        )
        assert _validation_failed(path) is True

    def test_mr_fail(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.json"
        path.write_text(
            json.dumps(
                {
                    "aggregate": {
                        "temporal_coherence": 0.85,
                        "keypoint_stability": 0.93,
                        "missing_rate": 0.06,
                    }
                }
            ),
            encoding="utf-8",
        )
        assert _validation_failed(path) is True


class TestLog:
    def test_log_creates_file_with_timestamp(self, tmp_path: Path) -> None:
        log_path = tmp_path / "pipeline.log"
        _log(log_path, "TEST_EVENT")
        content = log_path.read_text(encoding="utf-8")
        assert "TEST_EVENT" in content
        # Verify ISO timestamp prefix
        ts = content.split()[0]
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        assert dt.tzinfo == timezone.utc


def _write_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics), encoding="utf-8")


_passing_metrics: dict[str, Any] = {
    "aggregate": {
        "temporal_coherence": 0.85,
        "keypoint_stability": 0.93,
        "missing_rate": 0.04,
    },
    "videos": [],
}


class TestRunPipelineHappyPath:
    def test_pipeline_happy_path(self, tmp_path: Path, monkeypatch) -> None:
        log_path = tmp_path / "pipeline.log"
        metrics_path = tmp_path / "validation_metrics.json"
        report_path = tmp_path / "report.md"
        data_root = tmp_path / "data"

        def fake_run(cmd: list[str], **kwargs):  # type: ignore[no-untyped-def]
            if any("validate_2d_pose.py" in arg for arg in cmd):
                _write_metrics(metrics_path, _passing_metrics)
            if any("generate_validation_report.py" in arg for arg in cmd):
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text("# Report\n", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        rc = run_pipeline(
            data_root=data_root,
            log_path=log_path,
            metrics_path=metrics_path,
            report_path=report_path,
        )
        assert rc == 0
        assert log_path.is_file()
        logs = log_path.read_text(encoding="utf-8")
        assert "PIPELINE_START" in logs
        assert "PIPELINE_COMPLETE" in logs
        assert metrics_path.is_file()
        assert report_path.is_file()


class TestSelfCorrection:
    def test_self_correction_loop(self, tmp_path: Path, monkeypatch) -> None:
        """Simulate initial validation failure, then success after self-correction."""
        log_path = tmp_path / "pipeline.log"
        metrics_path = tmp_path / "validation_metrics.json"
        report_path = tmp_path / "report.md"
        data_root = tmp_path / "data"

        # Pre-fill failing metrics
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(
                {
                    "aggregate": {
                        "temporal_coherence": 0.80,
                        "keypoint_stability": 0.93,
                        "missing_rate": 0.04,
                    },
                    "videos": [],
                }
            ),
            encoding="utf-8",
        )

        call_log: list[list[str]] = []

        def fake_run(cmd: list[str], **kwargs):  # type: ignore[no-untyped-def]
            call_log.append(cmd)
            # Write passing metrics when re-validation runs after self-correction
            if any("validate_2d_pose.py" in arg for arg in cmd):
                _write_metrics(metrics_path, _passing_metrics)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        rc = run_pipeline(
            data_root=data_root,
            log_path=log_path,
            metrics_path=metrics_path,
            report_path=report_path,
            skip_validation=True,  # keep pre-filled metrics so initial failure is detected
        )

        # Phase 01 scripts must have been invoked
        detect_calls = [c for c in call_log if any("detect_anomalies_2d.py" in arg for arg in c)]
        correct_calls = [c for c in call_log if any("correct_trajectories_2d.py" in arg for arg in c)]
        assert len(detect_calls) >= 1
        assert len(correct_calls) >= 1
        # --strict-mode passed to correction
        assert any("--strict-mode" in arg for arg in correct_calls[0])

        # Check log content
        logs = log_path.read_text(encoding="utf-8")
        assert "VALIDATION_FAILED triggering self-correction" in logs
        assert "SELF_CORRECTION_SUCCEEDED" in logs
        assert "PIPELINE_COMPLETE" in logs
        assert rc == 0

    def test_self_correction_exhausted(self, tmp_path: Path, monkeypatch) -> None:
        """Simulate validation failure that persists after self-correction."""
        log_path = tmp_path / "pipeline.log"
        metrics_path = tmp_path / "validation_metrics.json"
        report_path = tmp_path / "report.md"
        data_root = tmp_path / "data"

        # Pre-fill failing metrics (this will still be failing after retry)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(
                {
                    "aggregate": {
                        "temporal_coherence": 0.80,
                        "keypoint_stability": 0.93,
                        "missing_rate": 0.04,
                    },
                    "videos": [],
                }
            ),
            encoding="utf-8",
        )

        def fake_run(cmd: list[str], **kwargs):  # type: ignore[no-untyped-def]
            # Re-validation still produces failing metrics
            if any("validate_2d_pose.py" in arg for arg in cmd):
                _write_metrics(
                    metrics_path,
                    {
                        "aggregate": {
                            "temporal_coherence": 0.80,
                            "keypoint_stability": 0.93,
                            "missing_rate": 0.04,
                        },
                        "videos": [],
                    },
                )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        rc = run_pipeline(
            data_root=data_root,
            log_path=log_path,
            metrics_path=metrics_path,
            report_path=report_path,
            skip_validation=True,  # metrics stay failing
        )

        logs = log_path.read_text(encoding="utf-8")
        assert "SELF_CORRECTION_EXHAUSTED validation still failing after retry" in logs
        assert rc == 1


class TestMainCLI:
    def test_main_invokes_pipeline(self, tmp_path: Path, monkeypatch) -> None:
        log_path = tmp_path / "pipeline.log"
        metrics_path = tmp_path / "validation_metrics.json"
        report_path = tmp_path / "report.md"

        def fake_run(cmd: list[str], **kwargs):  # type: ignore[no-untyped-def]
            if any("validate_2d_pose.py" in arg for arg in cmd):
                _write_metrics(metrics_path, _passing_metrics)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        # Override paths by calling run_pipeline directly; main uses fixed paths.
        rc = run_pipeline(
            data_root=tmp_path / "data",
            log_path=log_path,
            metrics_path=metrics_path,
            report_path=report_path,
            skip_training=True,
            skip_validation=False,
        )
        assert rc == 0
        assert metrics_path.is_file()
