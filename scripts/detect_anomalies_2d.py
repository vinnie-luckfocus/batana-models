#!/usr/bin/env python3
"""2D trajectory anomaly detection pipeline.

Reads per-video keypoint JSON, detects jump frames via displacement statistics
and optical flow validation for bat keypoints, and outputs anomaly reports.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy import stats
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_WINDOW_SIZE = 15
_JUMP_SIGMA_MULTIPLIER = 3.0
_BAT_FLOW_DEVIATION_THRESHOLD_PX = 20.0
_ACCEL_FALLBACK_THRESHOLD_PX = 40.0

_BAT_KEYPOINT_IDS = {9, 10, 17, 18}  # left_wrist, right_wrist, bat_knob, bat_barrel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def resolve_paths(data_root: Path) -> dict[str, Path]:
    """Return canonical paths for anomaly detection I/O."""
    return {
        "train_in": data_root / "train_2d_keypoints",
        "val_in": data_root / "val_2d_keypoints",
        "train_out": data_root / "train_2d_anomalies",
        "val_out": data_root / "val_2d_anomalies",
    }


def collect_json_files(directory: Path) -> list[Path]:
    """Collect all JSON files under a directory."""
    if not directory.exists():
        return []
    return sorted(p for p in directory.rglob("*.json") if p.is_file())


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_keypoint_data(path: Path) -> dict[str, Any]:
    """Load keypoint JSON."""
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


# ---------------------------------------------------------------------------
# Statistical anomaly detection
# ---------------------------------------------------------------------------


def compute_sliding_statistics(
    displacements: np.ndarray, window_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-frame mu and sigma using a sliding window over displacements.
    The center point is excluded so anomalies do not inflate their own threshold."""
    n = len(displacements)
    mu = np.full(n, np.nan, dtype=np.float32)
    sigma = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        start = max(0, i - window_size // 2)
        end = min(n, start + window_size)
        start = max(0, end - window_size)
        # Exclude the center point (index i) from the window
        window = displacements[start:end]
        mask = np.ones(len(window), dtype=bool)
        if start <= i < end:
            mask[i - start] = False
        valid = window[mask & ~np.isnan(window)]
        if len(valid) > 1:
            mu[i] = float(np.mean(valid))
            sigma[i] = float(np.std(valid, ddof=1))
    return mu, sigma


def detect_jump_frames(
    trajectory: np.ndarray, window_size: int = _WINDOW_SIZE, sigma_mult: float = _JUMP_SIGMA_MULTIPLIER
) -> list[int]:
    """Return frame indices where displacement exceeds mu + 3*sigma."""
    if len(trajectory) < 2:
        return []
    diffs = np.diff(trajectory, axis=0)
    displacements = np.linalg.norm(diffs, axis=1)
    mu, sigma = compute_sliding_statistics(displacements, window_size)

    jump_frames = []
    for i in range(len(displacements)):
        if np.isnan(mu[i]) or np.isnan(sigma[i]):
            continue
        threshold = mu[i] + sigma_mult * sigma[i]
        if threshold > 0 and displacements[i] > threshold:
            jump_frames.append(int(i + 1))  # displacement i corresponds to frame i+1
    return jump_frames


def detect_acceleration_fallback(trajectory: np.ndarray, threshold: float = _ACCEL_FALLBACK_THRESHOLD_PX) -> list[int]:
    """Return frame indices where second-order difference (acceleration) exceeds threshold."""
    if len(trajectory) < 3:
        return []
    accel = np.zeros(len(trajectory) - 2, dtype=np.float32)
    for i in range(2, len(trajectory)):
        if np.any(np.isnan(trajectory[i])) or np.any(np.isnan(trajectory[i - 1])) or np.any(np.isnan(trajectory[i - 2])):
            accel[i - 2] = 0.0
            continue
        a = np.linalg.norm(trajectory[i] - 2.0 * trajectory[i - 1] + trajectory[i - 2])
        accel[i - 2] = a
    return [int(i + 2) for i, a in enumerate(accel) if a > threshold]


# ---------------------------------------------------------------------------
# Optical flow validation for bat keypoints
# ---------------------------------------------------------------------------


def extract_keypoint_region(
    gray: np.ndarray, pt: np.ndarray, radius: int = 15
) -> np.ndarray:
    """Extract a square region around a point, padded if near edges."""
    h, w = gray.shape
    x, y = int(round(pt[0])), int(round(pt[1]))
    x1, x2 = max(0, x - radius), min(w, x + radius)
    y1, y2 = max(0, y - radius), min(h, y + radius)
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return np.zeros((radius * 2, radius * 2), dtype=np.uint8)
    # Pad to exact size if needed
    pad_bottom = max(0, radius * 2 - roi.shape[0])
    pad_right = max(0, radius * 2 - roi.shape[1])
    if pad_bottom or pad_right:
        roi = cv2.copyMakeBorder(roi, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=0)
    return roi


def compute_optical_flow_deviation(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    prev_pt: np.ndarray,
    curr_pt: np.ndarray,
) -> float | None:
    """Use Lucas-Kanade to predict next position from previous region and compute deviation."""
    if prev_gray is None or curr_gray is None:
        return None
    prev_roi = extract_keypoint_region(prev_gray, prev_pt, radius=15)
    curr_roi = extract_keypoint_region(curr_gray, prev_pt, radius=15)

    if prev_roi.shape != curr_roi.shape or prev_roi.size == 0:
        return None

    # Sparse LK on the full grayscale images centered around prev_pt
    p0 = np.array([[prev_pt[0], prev_pt[1]]], dtype=np.float32)
    p1, st, _err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None)
    if p1 is None or st is None or st[0][0] == 0:
        return None
    predicted = p1[0][0]
    deviation = float(np.linalg.norm(predicted - curr_pt))
    return deviation


def detect_bat_flow_anomalies(frames_data: dict[str, Any], video_path: Path | None = None) -> list[dict[str, Any]]:
    """Detect anomalies for bat keypoints using optical flow.

    If video_path is not provided, tries to resolve it from the preprocessed directory.
    """
    anomalies = []
    frames = frames_data.get("frames", [])
    if not frames:
        return anomalies

    # Resolve video path heuristically
    if video_path is None:
        data_root = Path("data")
        preprocessed_dir = data_root / "preprocessed"
        video_name = frames_data.get("video_name", "")
        candidate = preprocessed_dir / (video_name + ".mp4")
        if not candidate.exists():
            candidate = preprocessed_dir / (video_name.replace("_2d", "_preprocessed") + ".mp4")
        video_path = candidate if candidate.exists() else None

    if video_path is None or not video_path.exists():
        # Cannot run optical flow without video; rely on jump/accel detection later
        return anomalies

    cap = cv2.VideoCapture(str(video_path))
    grays = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()

    if len(grays) < 2 or len(grays) != len(frames):
        return anomalies

    for kid in _BAT_KEYPOINT_IDS:
        traj = extract_keypoint_trajectory(frames, kid)
        if len(traj) < 2:
            continue
        for i in range(1, len(frames)):
            if np.any(np.isnan(traj[i - 1])) or np.any(np.isnan(traj[i])):
                continue
            deviation = compute_optical_flow_deviation(grays[i - 1], grays[i], traj[i - 1], traj[i])
            if deviation is not None and deviation > _BAT_FLOW_DEVIATION_THRESHOLD_PX:
                anomalies.append(
                    {
                        "frame_id": i,
                        "keypoint_id": kid,
                        "anomaly_type": "OPTICAL_FLOW_DEVIATION",
                        "value": round(deviation, 2),
                    }
                )
    return anomalies


# ---------------------------------------------------------------------------
# Per-video pipeline
# ---------------------------------------------------------------------------


def process_video(json_path: Path) -> dict[str, Any]:
    """Run anomaly detection on a single video's keypoint JSON."""
    record: dict[str, Any] = {
        "video_name": json_path.stem,
        "source": str(json_path),
        "anomalies": [],
    }

    try:
        data = load_keypoint_data(json_path)
    except Exception as exc:
        record["error"] = str(exc)
        return record

    frames = data.get("frames", [])
    record["video_name"] = data.get("video_name", json_path.stem)

    # Determine number of keypoints from first frame
    num_keypoints = 0
    if frames:
        num_keypoints = max((kp["id"] for kp in frames[0].get("keypoints", [])), default=-1) + 1

    all_anomalies: list[dict[str, Any]] = []

    # 1) Statistical jump detection per keypoint
    for kid in range(num_keypoints):
        traj = extract_keypoint_trajectory(frames, kid)
        if len(traj) < 2:
            continue
        jump_frames = detect_jump_frames(traj)
        if not jump_frames:
            # Fallback to acceleration if statistics unreliable or no jumps found
            stats_usable = not np.all(np.isnan(traj))
            if not stats_usable:
                jump_frames = detect_acceleration_fallback(traj)
        for frame_id in jump_frames:
            all_anomalies.append(
                {
                    "frame_id": frame_id,
                    "keypoint_id": kid,
                    "anomaly_type": "JUMP_FRAME",
                    "value": None,
                }
            )

    # 2) Optical flow validation for bat keypoints
    bat_anomalies = detect_bat_flow_anomalies(data)
    all_anomalies.extend(bat_anomalies)

    # Deduplicate exact duplicates
    seen = set()
    deduped = []
    for a in all_anomalies:
        key = (a["frame_id"], a["keypoint_id"], a["anomaly_type"])
        if key not in seen:
            seen.add(key)
            deduped.append(a)

    record["anomalies"] = deduped
    record["anomaly_count"] = len(deduped)
    return record


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_anomaly_report(output_path: Path, record: dict[str, Any]) -> None:
    """Write anomaly report JSON for a single video."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Detect 2D trajectory anomalies")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args(argv)

    paths = resolve_paths(args.data_root)
    json_files = collect_json_files(paths["train_in"]) + collect_json_files(paths["val_in"])

    if not json_files:
        print("No keypoint JSON files found.")
        return 0

    for json_path in tqdm(json_files, desc="Detecting anomalies"):
        record = process_video(json_path)
        if "train" in str(json_path).lower() or "train" in json_path.parent.name:
            out_dir = paths["train_out"]
        else:
            out_dir = paths["val_out"]
        out_path = out_dir / (json_path.stem + "_anomalies.json")
        write_anomaly_report(out_path, record)

    print(f"Anomaly detection complete. Reports written to {paths['train_out']} and {paths['val_out']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
