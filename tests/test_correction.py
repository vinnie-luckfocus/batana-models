import numpy as np


def test_cubic_spline_single_gap():
    from scripts.correct_trajectories_2d import cubic_spline_fill
    traj = np.zeros((30, 2), dtype=np.float32)
    traj[:, 0] = np.arange(30, dtype=np.float32)
    # introduce a single gap at frame 10
    traj[10] = [100, 0]
    corrected = cubic_spline_fill(traj.copy(), 10, 10)
    # the corrected value should not be 100 anymore
    assert corrected[10, 0] != 100.0


def test_trajectory_consistency_smooth():
    from scripts.correct_trajectories_2d import compute_trajectory_consistency
    traj = np.zeros((30, 2), dtype=np.float32)
    traj[:, 0] = np.arange(30, dtype=np.float32)
    tc = compute_trajectory_consistency(traj)
    assert 0.0 <= tc <= 1.0


def test_tv_l2_smooth():
    from scripts.correct_trajectories_2d import tv_l2_smooth
    traj = np.zeros((30, 2), dtype=np.float32)
    traj[10] = [100, 0]
    smoothed = tv_l2_smooth(traj.copy())
    assert smoothed.shape == traj.shape
    assert smoothed[10, 0] != 100.0
