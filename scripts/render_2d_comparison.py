#!/usr/bin/env python3
"""Render side-by-side 2D comparison videos: raw vs corrected skeletons.

Requirements:
- 1920x1080 @ 30fps
- Left: raw skeleton in red
- Right: corrected skeleton in green, corrected keypoints highlighted in yellow circle
- Top overlay: video_id | Frame: N/120
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080
FPS = 30

CIRCLE_RADIUS = 6
CIRCLE_THICKNESS = 2
LINE_THICKNESS = 2

COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def resolve_paths(data_root: Path) -> dict[str, Path]:
    """Return canonical paths for rendering I/O."""
    return {
        "preprocessed": data_root / "preprocessed",
        "train_raw": data_root / "train_2d_keypoints",
        "val_raw": data_root / "val_2d_keypoints",
        "train_cleaned": data_root / "train_2d_cleaned",
        "val_cleaned": data_root / "val_2d_cleaned",
        "train_visuals": data_root / "visuals" / "2d" / "train",
        "val_visuals": data_root / "visuals" / "2d" / "val",
    }


def collect_json_files(directory: Path) -> list[Path]:
    """Collect all JSON files under a directory."""
    if not directory.exists():
        return []
    return sorted(p for p in directory.rglob("*.json") if p.is_file())


# ---------------------------------------------------------------------------
# Skeleton data loading
# ---------------------------------------------------------------------------


def load_keypoint_data(path: Path) -> dict[str, Any]:
    """Load keypoint JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_video_frames(video_path: Path) -> list[np.ndarray]:
    """Load all frames from a video."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def draw_skeleton(
    canvas: np.ndarray,
    keypoints: list[dict[str, Any]],
    color: tuple[int, int, int],
    highlight_color: tuple[int, int, int] | None = None,
    corrected_ids: set[int] | None = None,
) -> np.ndarray:
    """Draw skeleton bones and keypoints on a canvas."""
    out = canvas.copy()
    pts: dict[int, tuple[int, int]] = {}
    for kp in keypoints:
        kid = int(kp["id"])
        x = int(round(float(kp["x"])))
        y = int(round(float(kp["y"])))
        pts[kid] = (x, y)

    # Draw bones (standard COCO + bat)
    skeleton_pairs = [
        (16, 14),
        (14, 12),
        (17, 15),
        (15, 13),
        (12, 13),
        (6, 12),
        (7, 13),
        (6, 7),
        (6, 8),
        (7, 9),
        (8, 10),
        (9, 11),
        (2, 3),
        (1, 2),
        (1, 3),
        (2, 4),
        (3, 5),
        (4, 6),
        (5, 7),
        (17, 18),
    ]
    for a, b in skeleton_pairs:
        if a in pts and b in pts:
            cv2.line(out, pts[a], pts[b], color, LINE_THICKNESS)

    # Draw keypoints
    for kid, (x, y) in pts.items():
        is_highlight = corrected_ids is not None and kid in corrected_ids
        pt_color = highlight_color if is_highlight else color
        cv2.circle(out, (x, y), CIRCLE_RADIUS, pt_color, -1)
        if is_highlight:
            cv2.circle(out, (x, y), CIRCLE_RADIUS + 2, highlight_color, CIRCLE_THICKNESS)
    return out


def build_overlay_text(video_id: str, frame_number: int, total_frames: int) -> str:
    """Build the top overlay text."""
    return f"{video_id} | Frame: {frame_number + 1}/{total_frames}"


def render_overlay(canvas: np.ndarray, text: str) -> np.ndarray:
    """Render white text at the top center of the canvas."""
    out = canvas.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    size = cv2.getTextSize(text, font, scale, thickness)[0]
    x = (out.shape[1] - size[0]) // 2
    y = 40
    cv2.putText(out, text, (x, y), font, scale, COLOR_WHITE, thickness, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Diff helpers to detect corrected keypoints
# ---------------------------------------------------------------------------


def compute_corrected_ids(
    raw_frame: dict[str, Any], cleaned_frame: dict[str, Any], tolerance: float = 1.0
) -> set[int]:
    """Return keypoint ids whose coordinates changed by more than tolerance pixels."""
    raw_map = {kp["id"]: (float(kp["x"]), float(kp["y"])) for kp in raw_frame.get("keypoints", [])}
    clean_map = {
        kp["id"]: (float(kp["x"]), float(kp["y"])) for kp in cleaned_frame.get("keypoints", [])
    }
    corrected = set()
    for kid, (rx, ry) in raw_map.items():
        if kid in clean_map:
            cx, cy = clean_map[kid]
            if abs(rx - cx) > tolerance or abs(ry - cy) > tolerance:
                corrected.add(kid)
    return corrected


# ---------------------------------------------------------------------------
# Rendering pipeline
# ---------------------------------------------------------------------------


def render_video(
    video_path: Path,
    raw_data: dict[str, Any],
    cleaned_data: dict[str, Any],
    out_path: Path,
) -> bool:
    """Render a single side-by-side comparison video."""
    raw_frames_data = raw_data.get("frames", [])
    cleaned_frames_data = cleaned_data.get("frames", [])
    if not raw_frames_data or not cleaned_frames_data:
        return False

    video_frames = load_video_frames(video_path)
    if not video_frames:
        return False

    # Use the shorter of the three sequences
    total = min(len(video_frames), len(raw_frames_data), len(cleaned_frames_data))
    if total == 0:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(FPS), (OUTPUT_WIDTH, OUTPUT_HEIGHT))

    video_id = cleaned_data.get("video_name", out_path.stem)

    for i in range(total):
        base = cv2.resize(video_frames[i], (OUTPUT_WIDTH // 2, OUTPUT_HEIGHT))
        left = draw_skeleton(base, raw_frames_data[i]["keypoints"], COLOR_RED)
        corrected_ids = compute_corrected_ids(raw_frames_data[i], cleaned_frames_data[i])
        right = draw_skeleton(
            base,
            cleaned_frames_data[i]["keypoints"],
            COLOR_GREEN,
            highlight_color=COLOR_YELLOW,
            corrected_ids=corrected_ids,
        )
        combined = np.hstack([left, right])
        overlay_text = build_overlay_text(video_id, i, total)
        final = render_overlay(combined, overlay_text)
        writer.write(final)

    writer.release()
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render 2D raw vs corrected comparison videos")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args(argv)

    paths = resolve_paths(args.data_root)
    cleaned_files = collect_json_files(paths["train_cleaned"]) + collect_json_files(paths["val_cleaned"])

    if not cleaned_files:
        print("No cleaned keypoint JSON files found.")
        return 0

    success_count = 0
    for cleaned_path in tqdm(cleaned_files, desc="Rendering comparisons"):
        is_train = "train" in str(cleaned_path).lower() or "train" in cleaned_path.parent.name
        raw_path = paths["train_raw"] / cleaned_path.name if is_train else paths["val_raw"] / cleaned_path.name
        if not raw_path.exists():
            print(f"Skipping {cleaned_path.name}: raw JSON not found")
            continue

        raw_data = load_keypoint_data(raw_path)
        cleaned_data = load_keypoint_data(cleaned_path)
        video_name = cleaned_data.get("video_name", cleaned_path.stem)

        # Resolve preprocessed video path
        preprocessed_dir = paths["preprocessed"]
        video_path = preprocessed_dir / (video_name + "_preprocessed.mp4")
        if not video_path.exists():
            video_path = preprocessed_dir / (cleaned_path.stem.replace("_2d", "_preprocessed") + ".mp4")
        if not video_path.exists():
            print(f"Skipping {cleaned_path.name}: preprocessed video not found")
            continue

        out_dir = paths["train_visuals"] if is_train else paths["val_visuals"]
        out_path = out_dir / (cleaned_path.stem + "_compare.mp4")
        ok = render_video(video_path, raw_data, cleaned_data, out_path)
        if ok:
            success_count += 1
        else:
            print(f"Failed to render {cleaned_path.name}")

    print(f"Rendered {success_count}/{len(cleaned_files)} comparison videos.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
