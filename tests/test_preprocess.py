import pytest

pytest.importorskip("cv2")
import numpy as np


def test_clahe_shape_preserved():
    from scripts.preprocess_videos import apply_clahe_and_normalize
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    out = apply_clahe_and_normalize(frame)
    assert out.shape == frame.shape


def test_resample_indices_length():
    from scripts.preprocess_videos import compute_resample_indices
    indices = compute_resample_indices(15.0, 30.0, 60)
    assert len(indices) == 120


def test_qc_low_fps():
    from scripts.preprocess_videos import qcq_check
    status, reasons = qcq_check({"fps": 20, "width": 1920, "height": 1080})
    assert status == "UNREFINABLE"
    assert "LOW_FPS" in reasons


def test_qc_low_res():
    from scripts.preprocess_videos import qcq_check
    status, reasons = qcq_check({"fps": 30, "width": 1280, "height": 720})
    assert status == "OK" or "LOW_RES" not in reasons
