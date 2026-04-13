import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.validate_2d_pose import (
    compute_keypoint_stability,
    compute_missing_rate,
    compute_temporal_coherence,
    main,
)


class TestTemporalCoherence:
    def test_perfectly_smooth_tc_is_one(self):
        # Constant velocity → zero acceleration → TC = 1
        keypoints = [
            [(0.0, 0.0), (1.0, 1.0)],
            [(1.0, 0.0), (2.0, 1.0)],
            [(2.0, 0.0), (3.0, 1.0)],
            [(3.0, 0.0), (4.0, 1.0)],
        ]
        tc = compute_temporal_coherence(keypoints)
        assert math.isclose(tc, 1.0, abs_tol=1e-9)

    def test_no_frames_defaults_to_one(self):
        assert compute_temporal_coherence([]) == 1.0

    def test_single_frame_defaults_to_one(self):
        assert compute_temporal_coherence([[(0.0, 0.0)]]) == 1.0

    def test_two_frames_defaults_to_one(self):
        assert compute_temporal_coherence([[(0.0, 0.0)], [(1.0, 1.0)]]) == 1.0

    def test_high_acceleration_lowers_tc(self):
        keypoints = [
            [(0.0, 0.0)],
            [(0.0, 0.0)],
            [(10.0, 0.0)],
        ]
        tc = compute_temporal_coherence(keypoints)
        assert tc < 1.0
        # mean accel = ||10 - 0 + 0|| = 10
        assert math.isclose(tc, 1.0 / 11.0, abs_tol=1e-9)


class TestKeypointStability:
    def test_perfectly_stable_is_one(self):
        keypoints = [
            [(0.0, 0.0), (1.0, 1.0)],
            [(0.0, 0.0), (1.0, 1.0)],
            [(0.0, 0.0), (1.0, 1.0)],
        ]
        ks = compute_keypoint_stability(keypoints)
        assert math.isclose(ks, 1.0, abs_tol=1e-9)

    def test_no_displacements_defaults_to_one(self):
        assert compute_keypoint_stability([]) == 1.0

    def test_single_frame_defaults_to_one(self):
        assert compute_keypoint_stability([[(0.0, 0.0)]]) == 1.0

    def test_jump_frame_detected(self):
        # Many tiny displacements make sigma small, so a moderate jump is detected.
        keypoints = [[(0.0, 0.0)] for _ in range(21)]
        keypoints.append([(5.0, 0.0)])  # jump at frame 21
        keypoints.append([(5.0, 0.0)])  # stay there
        ks = compute_keypoint_stability(keypoints)
        # 22 displacement frames, 1 jump frame
        assert math.isclose(ks, 21 / 22, abs_tol=1e-9)

    def test_detectable_jump_ks(self):
        # Build many nearly-static frames so sigma is small, then one big jump
        keypoints = [[(0.0, 0.0)] for _ in range(21)]
        keypoints.append([(10.0, 0.0)])  # big jump at frame 21
        keypoints.append([(10.0, 0.0)])  # stay there
        ks = compute_keypoint_stability(keypoints)
        # 22 displacement frames, 1 jump frame
        assert math.isclose(ks, 21 / 22, abs_tol=1e-9)


class TestMissingRate:
    def test_no_missing(self):
        confidences = [[0.5, 0.9, 1.0]]
        assert compute_missing_rate(confidences) == 0.0

    def test_all_missing(self):
        confidences = [[0.0, 0.1, 0.29]]
        assert compute_missing_rate(confidences) == 1.0

    def test_mixed(self):
        confidences = [[0.5, 0.1, 0.0]]
        assert compute_missing_rate(confidences) == pytest.approx(2 / 3)

    def test_empty(self):
        assert compute_missing_rate([]) == 0.0


class TestEndToEnd:
    def test_dummy_validation_outputs_json(self, tmp_path: Path) -> None:
        output_path = tmp_path / "validation_metrics.json"
        exit_code = main(["--val-dir", str(tmp_path / "val"), "--output", str(output_path)])
        assert output_path.is_file()
        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert "videos" in data
        assert "aggregate" in data
        agg = data["aggregate"]
        assert "temporal_coherence" in agg
        assert "keypoint_stability" in agg
        assert "missing_rate" in agg
        # Dummy data should meet targets (fixed seed makes this deterministic)
        assert agg["temporal_coherence"] > 0.82
        assert agg["keypoint_stability"] > 0.92
        assert agg["missing_rate"] < 0.05
        assert exit_code == 0

    def test_cli_invocation(self, tmp_path: Path) -> None:
        script = Path(__file__).parent.parent / "scripts" / "validate_2d_pose.py"
        output_path = tmp_path / "metrics.json"
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--val-dir",
                str(tmp_path / "val"),
                "--output",
                str(output_path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert "aggregate" in data
