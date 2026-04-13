import numpy as np
import pytest


def test_detect_jumps_basic():
    from scripts.detect_anomalies_2d import detect_jump_frames
    trajectory = np.zeros((30, 2), dtype=np.float32)
    trajectory[10] = [500, 500]  # big jump
    jumps = detect_jump_frames(trajectory, window_size=15)
    assert 10 in jumps


def test_acceleration_fallback():
    from scripts.detect_anomalies_2d import detect_acceleration_fallback
    trajectory = np.zeros((30, 2), dtype=np.float32)
    trajectory[10] = [50, 50]
    jumps = detect_acceleration_fallback(trajectory, threshold=40.0)
    assert 10 in jumps
