import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_validation_report import generate_report, main


def test_report_contains_header():
    metrics = {
        "aggregate": {
            "temporal_coherence": 0.85,
            "keypoint_stability": 0.93,
            "missing_rate": 0.04,
        },
        "videos": [],
    }
    report = generate_report(metrics)
    assert "# 2D Pose Validation Report" in report


def test_report_contains_traceability():
    metrics = {
        "aggregate": {"temporal_coherence": 0.85, "keypoint_stability": 0.93, "missing_rate": 0.04},
        "videos": [],
    }
    report = generate_report(metrics, config_path="my_config.py", checkpoint_path="my_model.pth")
    assert "my_config.py" in report
    assert "my_model.pth" in report


def test_report_aggregate_pass_fail():
    # All pass
    report = generate_report({
        "aggregate": {"temporal_coherence": 0.85, "keypoint_stability": 0.93, "missing_rate": 0.04},
        "videos": [],
    })
    assert report.count("PASS") == 3
    assert report.count("FAIL") == 0

    # TC fails
    report = generate_report({
        "aggregate": {"temporal_coherence": 0.80, "keypoint_stability": 0.93, "missing_rate": 0.04},
        "videos": [],
    })
    assert report.count("PASS") == 2
    assert report.count("FAIL") == 1

    # KS fails (TC still passes, MR still passes)
    report = generate_report({
        "aggregate": {"temporal_coherence": 0.85, "keypoint_stability": 0.90, "missing_rate": 0.04},
        "videos": [],
    })
    assert report.count("PASS") == 2
    assert report.count("FAIL") == 1

    # MR fails (TC and KS still pass)
    report = generate_report({
        "aggregate": {"temporal_coherence": 0.85, "keypoint_stability": 0.93, "missing_rate": 0.06},
        "videos": [],
    })
    assert report.count("PASS") == 2
    assert report.count("FAIL") == 1


def test_report_per_video_table():
    metrics = {
        "aggregate": {"temporal_coherence": 0.85, "keypoint_stability": 0.93, "missing_rate": 0.04},
        "videos": [
            {"video_id": "vid_001", "frames": 30, "temporal_coherence": 0.84, "keypoint_stability": 0.94, "missing_rate": 0.03},
            {"video_id": "vid_002", "frames": 25, "temporal_coherence": 0.86, "keypoint_stability": 0.92, "missing_rate": 0.05},
        ],
    }
    report = generate_report(metrics)
    assert "| Video | Frames | TC | KS | Missing Rate |" in report
    assert "vid_001" in report
    assert "vid_002" in report
    assert "0.840000" in report


def test_report_timestamp_is_valid_iso():
    metrics = {
        "aggregate": {"temporal_coherence": 0.85, "keypoint_stability": 0.93, "missing_rate": 0.04},
        "videos": [],
    }
    report = generate_report(metrics)
    # Extract timestamp line
    for line in report.splitlines():
        if line.startswith("**Generated:**"):
            ts = line.replace("**Generated:**", "").strip()
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            assert dt.tzinfo == timezone.utc
            break
    else:
        pytest.fail("Timestamp not found in report")


def test_deterministic_output():
    metrics = {
        "aggregate": {"temporal_coherence": 0.85, "keypoint_stability": 0.93, "missing_rate": 0.04},
        "videos": [
            {"video_id": "vid_001", "frames": 30, "temporal_coherence": 0.84, "keypoint_stability": 0.94, "missing_rate": 0.03},
        ],
    }
    report1 = generate_report(metrics)
    report2 = generate_report(metrics)
    assert report1 == report2


class TestEndToEnd:
    def test_main_creates_report(self, tmp_path: Path) -> None:
        metrics = {
            "aggregate": {"temporal_coherence": 0.85, "keypoint_stability": 0.93, "missing_rate": 0.04},
            "videos": [
                {"video_id": "vid_001", "frames": 30, "temporal_coherence": 0.84, "keypoint_stability": 0.94, "missing_rate": 0.03},
            ],
        }
        metrics_path = tmp_path / "metrics.json"
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        output_path = tmp_path / "report.md"

        exit_code = main(["--metrics-json", str(metrics_path), "--output", str(output_path)])
        assert exit_code == 0
        assert output_path.is_file()
        content = output_path.read_text(encoding="utf-8")
        assert "# 2D Pose Validation Report" in content
        assert "vid_001" in content
        assert "PASS" in content

    def test_cli_invocation(self, tmp_path: Path) -> None:
        metrics = {
            "aggregate": {"temporal_coherence": 0.85, "keypoint_stability": 0.93, "missing_rate": 0.04},
            "videos": [],
        }
        metrics_path = tmp_path / "metrics.json"
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        output_path = tmp_path / "report.md"
        script = Path(__file__).parent.parent / "scripts" / "generate_validation_report.py"

        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--metrics-json",
                str(metrics_path),
                "--output",
                str(output_path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        content = output_path.read_text(encoding="utf-8")
        assert "# 2D Pose Validation Report" in content

    def test_main_missing_metrics(self, tmp_path: Path) -> None:
        exit_code = main(["--metrics-json", str(tmp_path / "missing.json")])
        assert exit_code == 1
