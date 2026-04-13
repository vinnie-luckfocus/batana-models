#!/usr/bin/env python3
"""2D trajectory auto-correction pipeline.

Reads raw 2D keypoints and anomaly reports, applies interpolation,
Kalman smoothing, optical flow + rigid bat model, and a global TV-L2 pass.
Writes cleaned keypoints and a correction log.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pykalman import KalmanFilter
from scipy.interpolate import CubicSpline
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TC_THRESHOLD = 0.82
_MAX_TV_ITERATIONS = 3
_TV_LAMBDA = 1.0
_BAT_LENGTH_TOLERANCE = 0.03

_SINGLE_FRAME_MAX = 1
_SHORT_GAP_MAX = 5
_LONG_GAP_MIN = 6

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def resolve_paths(data_root: Path) -> dict[str, Path]:
    """Return canonical paths for correction I/O."""
    return {
        "train_raw": data_root / "train_2d_keypoints",
        "val_raw": data_root / "val_2d_keypoints",
        "train_anomalies": data_root / "train_2d_anomalies",
        "val_anomalies": data_root / "val_2d_anomalies",
        "train_out": data_root / "train_2d_cleaned",
        "val_out": data_root / "val_2d_cleaned",
        "logs": data_root / "logs",
    }


def collect_json_files(directory: Path) -> list[Path]:
    """Collect all JSON files under a directory."""
    if not directory.exists():
        return []
    return sorted(p for p in directory.rglob("*.json") if p.is_file())


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_keypoint_trajectory(
    frames: list[dict[str, Any]], keypoint_id: int
) -> np.ndarray:
    """Extract (T, 2) trajectory for a single keypoint id."""
    pts = []
    for frame in frames:
        found = False
        for kp in frame.get("keypoints", []):
            if kp["id"] == keypoint_id:
                pts.append([kp["x"], kp["y"]])
                found = True
                break
        if not found:
            pts.append([np.nan, np.nan])
    return np.array(pts, dtype=np.float32)


def insert_trajectory_back(
    frames: list[dict[str, Any]], keypoint_id: int, trajectory: np.ndarray
) -> list[dict[str, Any]]:
    """Return a new frames list with updated keypoint coordinates."""
    new_frames = []
    for i, frame in enumerate(frames):
        new_kps = []
        updated = False
        for kp in frame.get("keypoints", []):
            if kp["id"] == keypoint_id:
                new_kps.append(
                    {
                        "id": kp["id"],
                        "x": float(trajectory[i, 0]),
                        "y": float(trajectory[i, 1]),
                        "confidence": kp.get("confidence", 0.0),
                    }
                )
                updated = True
            else:
                new_kps.append(dict(kp))
        if not updated and i < len(trajectory):
            new_kps.append(
                {
                    "id": keypoint_id,
                    "x": float(trajectory[i, 0]),
                    "y": float(trajectory[i, 1]),
                    "confidence": 0.0,
                }
            )
        new_frames.append({"frame_id": frame.get("frame_id", i), "keypoints": new_kps})
    return new_frames


# ---------------------------------------------------------------------------
# Gap identification
# ---------------------------------------------------------------------------


def find_anomaly_gaps(
    trajectory: np.ndarray, anomalies: list[dict[str, Any]], keypoint_id: int
) -> list[tuple[int, int]]:
    """Return contiguous anomaly frame ranges for a given keypoint."""
    frames = sorted(
        {a["frame_id"] for a in anomalies if a["keypoint_id"] == keypoint_id}
    )
    if not frames:
        return []
    gaps: list[tuple[int, int]] = []
    start = frames[0]
    prev = frames[0]
    for f in frames[1:]:
        if f == prev + 1:
            prev = f
        else:
            gaps.append((start, prev))
            start = f
            prev = f
    gaps.append((start, prev))
    return gaps


# ---------------------------------------------------------------------------
# Correction methods
# ---------------------------------------------------------------------------


def _valid_neighborhood(trajectory: np.ndarray, center: int, radius: int) -> np.ndarray:
    """Return valid points within radius of center."""
    start = max(0, center - radius)
    end = min(len(trajectory), center + radius + 1)
    pts = trajectory[start:end]
    valid_mask = ~np.isnan(pts[:, 0]) & ~np.isnan(pts[:, 1])
    return pts[valid_mask]


def cubic_spline_fill(trajectory: np.ndarray, gap_start: int, gap_end: int) -> np.ndarray:
    """Fill a single-frame gap using cubic spline with 5 neighbors each side."""
    traj = trajectory.copy()
    n = len(traj)
    # Gather valid control points
    left_idx = []
    left_vals = []
    for i in range(gap_start - 1, -1, -1):
        if not np.isnan(traj[i, 0]) and not np.isnan(traj[i, 1]):
            left_idx.append(i)
            left_vals.append(traj[i].copy())
        if len(left_idx) >= 5:
            break
    right_idx = []
    right_vals = []
    for i in range(gap_end + 1, n):
        if not np.isnan(traj[i, 0]) and not np.isnan(traj[i, 1]):
            right_idx.append(i)
            right_vals.append(traj[i].copy())
        if len(right_idx) >= 5:
            break

    if len(left_idx) + len(right_idx) < 4:
        # Not enough control points; linear interpolation fallback
        return linear_fill(traj, gap_start, gap_end)

    x_nodes = np.array(left_idx[::-1] + right_idx, dtype=np.float32)
    y_nodes_x = np.array([v[0] for v in left_vals[::-1] + right_vals], dtype=np.float32)
    y_nodes_y = np.array([v[1] for v in left_vals[::-1] + right_vals], dtype=np.float32)

    cs_x = CubicSpline(x_nodes, y_nodes_x)
    cs_y = CubicSpline(x_nodes, y_nodes_y)
    for t in range(gap_start, gap_end + 1):
        traj[t, 0] = float(cs_x(t))
        traj[t, 1] = float(cs_y(t))
    return traj


def linear_fill(trajectory: np.ndarray, gap_start: int, gap_end: int) -> np.ndarray:
    """Linear interpolation fallback for a gap."""
    traj = trajectory.copy()
    n = len(traj)
    prev_valid = None
    for i in range(gap_start - 1, -1, -1):
        if not np.isnan(traj[i, 0]):
            prev_valid = i
            break
    next_valid = None
    for i in range(gap_end + 1, n):
        if not np.isnan(traj[i, 0]):
            next_valid = i
            break
    if prev_valid is None and next_valid is None:
        return traj
    if prev_valid is None:
        traj[gap_start : gap_end + 1] = traj[next_valid]
        return traj
    if next_valid is None:
        traj[gap_start : gap_end + 1] = traj[prev_valid]
        return traj
    alpha = np.linspace(0.0, 1.0, gap_end - gap_start + 2)[1:-1]
    for j, a in enumerate(alpha):
        traj[gap_start + j] = (1 - a) * traj[prev_valid] + a * traj[next_valid]
    return traj


def kalman_smooth_fill(trajectory: np.ndarray, gap_start: int, gap_end: int) -> np.ndarray:
    """Fill a 2-5 frame gap using Kalman Smoother (RTS) with constant-velocity model."""
    traj = trajectory.copy()
    n = len(traj)
    # Mark gap as missing
    observations = traj.copy()
    for t in range(gap_start, gap_end + 1):
        observations[t] = np.nan

    # Collect valid observations
    valid_mask = ~np.isnan(observations[:, 0])
    if valid_mask.sum() < 3:
        # Fallback to linear
        return linear_fill(traj, gap_start, gap_end)

    valid_times = np.where(valid_mask)[0]
    valid_obs = observations[valid_mask]

    # Constant velocity state: [x, y, vx, vy]
    kf = KalmanFilter(
        transition_matrices=np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        ),
        observation_matrices=np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        ),
        initial_state_mean=np.array(
            [valid_obs[0, 0], valid_obs[0, 1], 0.0, 0.0], dtype=np.float32
        ),
        n_dim_obs=2,
    )

    # EM on valid observations (using the first valid sequence as-is)
    # pykalman smooth works on full sequences with NaNs, but EM doesn't handle NaNs
    kf = kf.em(valid_obs, n_iter=2)

    # Smooth full sequence
    smoothed_state_means, _ = kf.smooth(observations)
    smoothed_positions = smoothed_state_means[:, :2]

    # Only fill gap region
    for t in range(gap_start, gap_end + 1):
        traj[t] = smoothed_positions[t]
    return traj


def optical_flow_rigid_bat_fill(
    trajectory: np.ndarray,
    gap_start: int,
    gap_end: int,
    video_path: Path | None,
) -> np.ndarray:
    """Fill a >5 frame gap using optical flow tracking + rigid bat length constraint."""
    traj = trajectory.copy()
    n = len(traj)

    # Load video frames
    if video_path is None or not video_path.exists():
        return linear_fill(traj, gap_start, gap_end)

    cap = cv2.VideoCapture(str(video_path))
    grays = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()

    if len(grays) != n:
        return linear_fill(traj, gap_start, gap_end)

    # Find nearest valid frame before and after gap
    prev_valid = None
    for i in range(gap_start - 1, -1, -1):
        if not np.isnan(traj[i, 0]):
            prev_valid = i
            break
    next_valid = None
    for i in range(gap_end + 1, n):
        if not np.isnan(traj[i, 0]):
            next_valid = i
            break

    if prev_valid is None and next_valid is None:
        return traj
    if prev_valid is None or next_valid is None:
        return linear_fill(traj, gap_start, gap_end)

    # Track using LK from prev_valid through gap
    p0 = np.array([[traj[prev_valid, 0], traj[prev_valid, 1]]], dtype=np.float32)
    tracked = [traj[prev_valid].copy()]
    for i in range(prev_valid + 1, next_valid + 1):
        p1, st, _err = cv2.calcOpticalFlowPyrLK(grays[i - 1], grays[i], p0, None)
        if p1 is None or st is None or st[0][0] == 0:
            # Fallback to linear for this step
            alpha = (i - prev_valid) / max(1, next_valid - prev_valid)
            pt = (1 - alpha) * traj[prev_valid] + alpha * traj[next_valid]
        else:
            pt = p1[0][0]
        tracked.append(pt.copy())
        p0 = np.array([pt], dtype=np.float32)

    # Rigid bat length constraint: compute median bat length from non-gap frames
    # For this generic function, we compute median displacement from prev_valid / next_valid
    lengths = []
    for i in range(n):
        if i != prev_valid and i != next_valid and not np.isnan(traj[i, 0]):
            # Use distance between consecutive valid frames as proxy
            lengths.append(float(np.linalg.norm(traj[i] - traj[i - 1])))
    median_length = float(np.median(lengths)) if lengths else 10.0

    # Apply tracked positions inside gap and enforce length constraint
    for idx, t in enumerate(range(prev_valid + 1, next_valid + 1)):
        if t < gap_start or t > gap_end:
            continue
        raw_pt = tracked[idx + 1]
        # Enforce that displacement from prev frame stays within +/- 3% of median proxy
        prev_pt = traj[t - 1] if t - 1 >= 0 else raw_pt
        disp = float(np.linalg.norm(raw_pt - prev_pt))
        min_d = median_length * (1 - _BAT_LENGTH_TOLERANCE)
        max_d = median_length * (1 + _BAT_LENGTH_TOLERANCE)
        if disp < min_d or disp > max_d:
            # Scale towards prev_pt to satisfy constraint
            direction = raw_pt - prev_pt
            dir_norm = float(np.linalg.norm(direction)) + 1e-6
            scaled = prev_pt + (direction / dir_norm) * np.clip(disp, min_d, max_d)
            raw_pt = scaled
        traj[t] = raw_pt

    return traj


# ---------------------------------------------------------------------------
# Global TV-L2 smoothing
# ---------------------------------------------------------------------------


def tv_l2_smooth(trajectory: np.ndarray, lambda_tv: float = _TV_LAMBDA, iterations: int = 10) -> np.ndarray:
    """Apply temporal total variation (TV-L2) smoothing using gradient descent.

    Minimizes sum_t ||P_{t+1} - 2P_t + P_{t-1}||^2 + lambda * data fidelity.
    """
    traj = trajectory.copy()
    n = traj.shape[0]
    mask = ~np.isnan(traj[:, 0]) & ~np.isnan(traj[:, 1])
    if mask.sum() < 3:
        return traj

    # Fill missing temporarily for TV pass
    filled = traj.copy()
    for i in range(n):
        if np.isnan(filled[i, 0]):
            # nearest valid
            distances = np.arange(n)
            valid_idx = np.where(mask)[0]
            if len(valid_idx) == 0:
                continue
            nearest = valid_idx[np.argmin(np.abs(valid_idx - i))]
            filled[i] = traj[nearest]

    # Simple iterative smoother
    for _ in range(iterations):
        new_filled = filled.copy()
        for t in range(1, n - 1):
            accel = filled[t + 1] - 2.0 * filled[t] + filled[t - 1]
            grad = 2.0 * accel
            # Data term pulls towards original if it was valid
            if mask[t]:
                grad = grad + lambda_tv * (filled[t] - traj[t])
            new_filled[t] = filled[t] - 0.1 * grad
        filled = new_filled

    # Restore original valid points partially to preserve data fidelity
    result = filled.copy()
    for i in range(n):
        if mask[i]:
            result[i] = 0.7 * traj[i] + 0.3 * filled[i]
    return result


def compute_trajectory_consistency(trajectory: np.ndarray) -> float:
    """Compute TC = 1 / (1 + mean(acceleration magnitude))."""
    if len(trajectory) < 3:
        return 0.0
    accels = []
    for i in range(2, len(trajectory)):
        if (
            np.isnan(trajectory[i, 0])
            or np.isnan(trajectory[i - 1, 0])
            or np.isnan(trajectory[i - 2, 0])
        ):
            continue
        a = float(np.linalg.norm(trajectory[i] - 2.0 * trajectory[i - 1] + trajectory[i - 2]))
        accels.append(a)
    if not accels:
        return 0.0
    mean_accel = float(np.mean(accels))
    return 1.0 / (1.0 + mean_accel)


# ---------------------------------------------------------------------------
# Per-video correction
# ---------------------------------------------------------------------------


def correct_video(
    raw_path: Path, anomaly_path: Path, out_path: Path, video_path: Path | None
) -> dict[str, Any]:
    """Apply full correction pipeline to a single video."""
    record: dict[str, Any] = {
        "video_name": raw_path.stem,
        "status": "PENDING",
        "corrections": [],
    }

    try:
        raw_data = load_json(raw_path)
        anomaly_data = load_json(anomaly_path)
    except Exception as exc:
        record["status"] = "ERROR"
        record["error"] = str(exc)
        return record

    frames = raw_data.get("frames", [])
    if not frames:
        record["status"] = "ERROR"
        record["error"] = "No frames in raw data"
        return record

    num_keypoints = 0
    if frames:
        num_keypoints = max((kp["id"] for kp in frames[0].get("keypoints", [])), default=-1) + 1

    anomalies = anomaly_data.get("anomalies", [])

    # Correct per keypoint
    corrected_frames = frames
    correction_log: list[dict[str, Any]] = []

    for kid in range(num_keypoints):
        traj = extract_keypoint_trajectory(corrected_frames, kid)
        gaps = find_anomaly_gaps(traj, anomalies, kid)

        for gap_start, gap_end in gaps:
            gap_len = gap_end - gap_start + 1
            if gap_len <= _SINGLE_FRAME_MAX:
                traj = cubic_spline_fill(traj, gap_start, gap_end)
                method = "cubic_spline"
            elif gap_len <= _SHORT_GAP_MAX:
                traj = kalman_smooth_fill(traj, gap_start, gap_end)
                method = "kalman_smoother"
            else:
                traj = optical_flow_rigid_bat_fill(traj, gap_start, gap_end, video_path)
                method = "optical_flow_rigid"
            correction_log.append(
                {
                    "keypoint_id": kid,
                    "gap_start": gap_start,
                    "gap_end": gap_end,
                    "method": method,
                }
            )

        # Global TV-L2 (iterate up to 3 times if TC < threshold)
        for iteration in range(_MAX_TV_ITERATIONS):
            traj = tv_l2_smooth(traj, lambda_tv=_TV_LAMBDA)
            tc = compute_trajectory_consistency(traj)
            if tc >= _TC_THRESHOLD:
                break
        else:
            tc = compute_trajectory_consistency(traj)

        corrected_frames = insert_trajectory_back(corrected_frames, kid, traj)
        record["corrections"].append(
            {
                "keypoint_id": kid,
                "methods": [c["method"] for c in correction_log if c["keypoint_id"] == kid],
                "final_tc": round(tc, 4),
            }
        )

    # Determine overall status
    overall_tc = min(
        (c["final_tc"] for c in record["corrections"]),
        default=0.0,
    )
    if overall_tc < _TC_THRESHOLD:
        record["status"] = "UNREFINABLE"
    else:
        record["status"] = "CLEANED"
    record["overall_tc"] = round(overall_tc, 4)

    # Write cleaned JSON
    out_payload = {
        "video_name": raw_data.get("video_name", raw_path.stem),
        "fps": raw_data.get("fps", 30.0),
        "resolution": raw_data.get("resolution", [1920, 1080]),
        "keypoint_schema": raw_data.get("keypoint_schema", "coco-17-plus-bat"),
        "frames": corrected_frames,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)

    return record


# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------


def append_to_csv(log_path: Path, records: list[dict[str, Any]]) -> None:
    """Append correction records to a CSV log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["video_name", "keypoint_id", "method", "final_tc", "overall_tc", "status"]
    rows = []
    for rec in records:
        base = {
            "video_name": rec["video_name"],
            "overall_tc": rec.get("overall_tc", ""),
            "status": rec["status"],
        }
        corrections = rec.get("corrections", [])
        if not corrections:
            rows.append({**base, "keypoint_id": "", "method": "", "final_tc": ""})
        else:
            for c in corrections:
                methods = c.get("methods", [])
                method_str = ";".join(methods) if methods else "none"
                rows.append(
                    {
                        **base,
                        "keypoint_id": c["keypoint_id"],
                        "method": method_str,
                        "final_tc": c["final_tc"],
                    }
                )

    write_header = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Auto-correct 2D trajectories")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--strict-mode", action="store_true", help="Use stricter correction thresholds")
    args = parser.parse_args(argv)

    if args.strict_mode:
        global _TC_THRESHOLD, _TV_LAMBDA, _BAT_LENGTH_TOLERANCE
        _TC_THRESHOLD = 0.85
        _TV_LAMBDA = 0.5
        _BAT_LENGTH_TOLERANCE = 0.01

    paths = resolve_paths(args.data_root)
    raw_files = collect_json_files(paths["train_raw"]) + collect_json_files(paths["val_raw"])

    if not raw_files:
        print("No raw keypoint JSON files found.")
        return 0

    all_records: list[dict[str, Any]] = []
    for raw_path in tqdm(raw_files, desc="Correcting trajectories"):
        anomaly_path = (
            paths["train_anomalies"] / (raw_path.stem + "_anomalies.json")
            if "train" in str(raw_path).lower() or "train" in raw_path.parent.name
            else paths["val_anomalies"] / (raw_path.stem + "_anomalies.json")
        )
        if not anomaly_path.exists():
            # Some files may not have anomaly reports if they had no anomalies
            anomaly_path.parent.mkdir(parents=True, exist_ok=True)
            anomaly_path.write_text(
                json.dumps({"video_name": raw_path.stem, "anomalies": []}), encoding="utf-8"
            )

        out_path = (
            paths["train_out"] / raw_path.name
            if "train" in str(raw_path).lower() or "train" in raw_path.parent.name
            else paths["val_out"] / raw_path.name
        )

        # Resolve preprocessed video path for optical flow
        preprocessed_dir = paths["train_out"].parent / "preprocessed"
        video_path = preprocessed_dir / (raw_path.stem.replace("_2d", "_preprocessed") + ".mp4")
        if not video_path.exists():
            video_path = preprocessed_dir / (raw_path.stem + "_preprocessed.mp4")
        if not video_path.exists():
            video_path = None

        record = correct_video(raw_path, anomaly_path, out_path, video_path)
        all_records.append(record)

    log_path = paths["logs"] / "2d_cleaning_log.csv"
    append_to_csv(log_path, all_records)

    print(
        f"Correction complete. Cleaned JSON written to {paths['train_out']} and {paths['val_out']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
