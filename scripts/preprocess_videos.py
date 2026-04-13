#!/usr/bin/env python3
"""Video preprocessing pipeline for baseball swing analysis.

Reads raw videos, applies QC checks, resamples, scales, enhances,
detects swing completeness, extracts core swing clips, and writes
preprocessed outputs plus a QC report.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants (from PRD constraints)
# ---------------------------------------------------------------------------
TARGET_FPS = 30
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
MIN_FPS = 24
MIN_RES_SHORT_SIDE = 720
TARGET_SHORT_SIDE = 1080
TARGET_FRAME_COUNT = 120

WRIST_VELOCITY_THRESHOLD_PX = 30.0
BAT_ORIENTATION_CHANGE_THRESHOLD_DEG = 15.0

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = 8

VIDEO_EXTS = {".mp4", ".mov"}

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def resolve_paths(data_root: Path) -> dict[str, Path]:
    """Return canonical paths for all pipeline directories."""
    return {
        "raw_train": data_root / "raw_videos" / "train",
        "raw_val": data_root / "raw_videos" / "val",
        "preprocessed": data_root / "preprocessed",
        "logs": data_root / "logs",
    }


def collect_video_files(directory: Path) -> list[Path]:
    """Collect all MP4/MOV files under a directory."""
    if not directory.exists():
        return []
    files = sorted(
        p for p in directory.rglob("*")
        if p.suffix.lower() in VIDEO_EXTS and p.is_file()
    )
    return files


# ---------------------------------------------------------------------------
# Video property reading
# ---------------------------------------------------------------------------

def read_video_properties(path: Path) -> dict[str, Any]:
    """Read fps, width, and height from a video file using OpenCV."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return {
        "fps": float(fps),
        "width": width,
        "height": height,
        "frame_count": frame_count,
    }


def qcq_check(props: dict[str, Any]) -> tuple[str, list[str]]:
    """Return QC status and list of flags based on video properties."""
    flags: list[str] = []

    if props["fps"] < MIN_FPS:
        flags.append("LOW_FPS")
    short_side = min(props["width"], props["height"])
    if short_side < MIN_RES_SHORT_SIDE:
        flags.append("LOW_RES")

    if flags:
        return "UNREFINABLE", flags
    return "PASS", flags


# ---------------------------------------------------------------------------
# Frame-level processing helpers
# ---------------------------------------------------------------------------

def compute_resample_indices(original_fps: float, target_fps: float, frame_count: int) -> list[int]:
    """Compute frame indices to keep when resampling via duplication/dropping."""
    if frame_count <= 0:
        return []
    ratio = target_fps / original_fps
    indices = []
    for i in range(int(math.ceil(frame_count * ratio))):
        src_idx = int(round(i / ratio))
        src_idx = min(src_idx, frame_count - 1)
        indices.append(src_idx)
    return indices


def scale_and_crop_frame(frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Scale short side to target_height and center-crop to target_width x target_height."""
    h, w = frame.shape[:2]
    short_side = min(h, w)
    scale = target_height / short_side
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Center crop
    crop_x = max(0, (new_w - target_width) // 2)
    crop_y = max(0, (new_h - target_height) // 2)
    cropped = resized[crop_y : crop_y + target_height, crop_x : crop_x + target_width]

    # Pad if necessary
    pad_bottom = target_height - cropped.shape[0]
    pad_right = target_width - cropped.shape[1]
    if pad_bottom > 0 or pad_right > 0:
        cropped = cv2.copyMakeBorder(
            cropped, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

    return cropped


def apply_clahe_and_normalize(frame: np.ndarray) -> np.ndarray:
    """Apply CLAHE in LAB space and normalize pixel values to [0, 255]."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(CLAHE_TILE_SIZE, CLAHE_TILE_SIZE))
    l_eq = clahe.apply(l)
    merged = cv2.merge([l_eq, a, b])
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # Normalize to full range
    norm = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    return norm


# ---------------------------------------------------------------------------
# Swing detection helpers
# ---------------------------------------------------------------------------

def extract_all_frames(path: Path) -> list[np.ndarray]:
    """Read all frames from a video file."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def detect_swing_segments(frames: list[np.ndarray]) -> list[tuple[int, int]]:
    """Detect candidate swing segments based on wrist velocity and bat orientation.

    Uses simple frame-differencing of wrist regions as a proxy for velocity,
    and bat orientation change computed from motion vectors in the lower-center region.
    """
    if len(frames) < 3:
        return []

    h, w = frames[0].shape[:2]

    def wrist_velocity_score(gray_t: np.ndarray, gray_t1: np.ndarray) -> float:
        # Focus on center-lower half where wrists typically move
        y_start = h // 2
        roi_t = gray_t[y_start:, :]
        roi_t1 = gray_t1[y_start:, :]
        diff = cv2.absdiff(roi_t, roi_t1)
        return float(np.mean(diff))

    def bat_orientation_change(gray_t: np.ndarray, gray_t1: np.ndarray) -> float:
        # Use optical flow magnitude in center region as proxy for orientation change
        y_start = h // 3
        x_start = w // 4
        roi_t = gray_t[y_start:, x_start : x_start + w // 2]
        roi_t1 = gray_t1[y_start:, x_start : x_start + w // 2]
        flow = cv2.calcOpticalFlowFarneback(
            roi_t, roi_t1, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return float(np.mean(mag))

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

    wrist_vels = []
    bat_oris = []
    for i in range(1, len(grays)):
        wrist_vels.append(wrist_velocity_score(grays[i - 1], grays[i]))
        bat_oris.append(bat_orientation_change(grays[i - 1], grays[i]))

    # Find peaks
    peak_frames: list[int] = []
    for i in range(1, len(wrist_vels) - 1):
        is_peak = (
            wrist_vels[i] > WRIST_VELOCITY_THRESHOLD_PX
            and wrist_vels[i] > wrist_vels[i - 1]
            and wrist_vels[i] > wrist_vels[i + 1]
        )
        ori_change = bat_oris[i] if i < len(bat_oris) else 0.0
        orientation_ok = ori_change > BAT_ORIENTATION_CHANGE_THRESHOLD_DEG
        if is_peak and orientation_ok:
            peak_frames.append(i)

    if not peak_frames:
        # Fallback: use region with highest combined score
        combined = [w + o for w, o in zip(wrist_vels, bat_oris)]
        max_idx = int(np.argmax(combined)) + 1
        peak_frames = [max_idx]

    segments: list[tuple[int, int]] = []
    half_length = TARGET_FRAME_COUNT // 2
    for peak in peak_frames:
        start = max(0, peak - half_length)
        end = min(len(frames), start + TARGET_FRAME_COUNT)
        start = max(0, end - TARGET_FRAME_COUNT)
        segments.append((start, end))

    # Deduplicate overlapping segments
    deduped: list[tuple[int, int]] = []
    for seg in sorted(segments):
        if not deduped or seg[0] >= deduped[-1][1]:
            deduped.append(seg)
    return deduped


def extract_core_swing(
    frames: list[np.ndarray],
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Extract the main swing segment and pad/trim to exactly TARGET_FRAME_COUNT."""
    segments = detect_swing_segments(frames)
    if not segments:
        # Fallback: take middle chunk
        start = max(0, (len(frames) - TARGET_FRAME_COUNT) // 2)
        end = min(len(frames), start + TARGET_FRAME_COUNT)
        chosen = frames[start:end]
        meta = {"swing_detected": False, "segment_start": start, "segment_end": end}
    else:
        start, end = segments[0]
        chosen = frames[start:end]
        meta = {"swing_detected": True, "segment_start": start, "segment_end": end}

    # Pad or trim
    result: list[np.ndarray] = []
    if len(chosen) >= TARGET_FRAME_COUNT:
        result = chosen[:TARGET_FRAME_COUNT]
    else:
        result = list(chosen)
        last = chosen[-1] if chosen else np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
        while len(result) < TARGET_FRAME_COUNT:
            result.append(last.copy())

    meta["output_frame_count"] = len(result)
    return result, meta


# ---------------------------------------------------------------------------
# Video writer
# ---------------------------------------------------------------------------

def write_video(frames: list[np.ndarray], out_path: Path, fps: int = TARGET_FPS) -> None:
    """Write a list of BGR frames to an MP4 file using H.264."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


# ---------------------------------------------------------------------------
# Per-video pipeline
# ---------------------------------------------------------------------------

def process_video(
    src_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Process a single raw video through the full pipeline."""
    record: dict[str, Any] = {
        "source": str(src_path),
        "status": "PENDING",
        "flags": [],
    }

    try:
        props = read_video_properties(src_path)
        record.update(props)
    except Exception as exc:
        record["status"] = "ERROR"
        record["error"] = str(exc)
        return record

    status, flags = qcq_check(props)
    record["flags"] = flags
    if status != "PASS":
        record["status"] = status
        return record

    try:
        frames = extract_all_frames(src_path)
    except Exception as exc:
        record["status"] = "ERROR"
        record["error"] = f"Frame extraction failed: {exc}"
        return record

    if not frames:
        record["status"] = "ERROR"
        record["error"] = "No frames extracted"
        return record

    # Resample to 30fps
    indices = compute_resample_indices(props["fps"], TARGET_FPS, len(frames))
    resampled = [frames[i].copy() for i in indices]

    # Scale, crop, enhance
    processed = []
    for frame in resampled:
        cropped = scale_and_crop_frame(frame, TARGET_WIDTH, TARGET_HEIGHT)
        enhanced = apply_clahe_and_normalize(cropped)
        processed.append(enhanced)

    # Extract core swing
    core_frames, swing_meta = extract_core_swing(processed)
    record["swing_meta"] = swing_meta

    rel_name = src_path.stem + "_preprocessed.mp4"
    out_path = out_dir / rel_name
    write_video(core_frames, out_path, TARGET_FPS)

    record["output_path"] = str(out_path)
    record["status"] = "USABLE"
    return record


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preprocess raw baseball swing videos")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args(argv)

    paths = resolve_paths(args.data_root)
    paths["preprocessed"].mkdir(parents=True, exist_ok=True)
    paths["logs"].mkdir(parents=True, exist_ok=True)

    raw_files = (
        collect_video_files(paths["raw_train"])
        + collect_video_files(paths["raw_val"])
    )

    qc_records: list[dict[str, Any]] = []
    usable_train = 0

    for src in tqdm(raw_files, desc="Preprocessing videos"):
        record = process_video(src, paths["preprocessed"])
        qc_records.append(record)
        if record["status"] == "USABLE" and "train" in record["source"]:
            usable_train += 1

    qc_report_path = paths["logs"] / "qc_report.json"
    with open(qc_report_path, "w") as f:
        json.dump(qc_records, f, indent=2)

    flag_path = paths["logs"] / "emergency_augmentation.flag"
    if usable_train < 50:
        flag_path.write_text(f"usable_train={usable_train}\n", encoding="utf-8")
        print(f"⚠️  Only {usable_train} usable training videos. Flag written: {flag_path}")
    else:
        if flag_path.exists():
            flag_path.unlink()

    summary = {
        "total": len(qc_records),
        "usable": sum(1 for r in qc_records if r["status"] == "USABLE"),
        "unrefinable": sum(1 for r in qc_records if r["status"] == "UNREFINABLE"),
        "errors": sum(1 for r in qc_records if r["status"] not in {"USABLE", "UNREFINABLE"}),
    }
    print(f"QC complete: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
