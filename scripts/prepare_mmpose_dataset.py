#!/usr/bin/env python3
"""Prepare MMPose COCO-format dataset from cleaned 2D keypoints.

Reads cleaned keypoint JSONs, extracts frames from preprocessed videos,
writes image files, and generates COCO-format annotations.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "bat_knob",
    "bat_barrel",
]

COCO_SKELETON = [
    [16, 14],
    [14, 12],
    [17, 15],
    [15, 13],
    [12, 13],
    [6, 12],
    [7, 13],
    [6, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [9, 11],
    [2, 3],
    [1, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
    [5, 7],
    [17, 18],
]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def resolve_paths(data_root: Path) -> dict[str, Path]:
    """Return canonical paths for dataset preparation I/O."""
    return {
        "train_cleaned": data_root / "train_2d_cleaned",
        "val_cleaned": data_root / "val_2d_cleaned",
        "preprocessed": data_root / "preprocessed",
        "images_train": data_root / "mmpose_baseball" / "images" / "train",
        "images_val": data_root / "mmpose_baseball" / "images" / "val",
        "annotations": data_root / "mmpose_baseball" / "annotations",
        "logs": data_root / "logs",
    }


# ---------------------------------------------------------------------------
# Video / frame helpers
# ---------------------------------------------------------------------------


def find_preprocessed_video(preprocessed_dir: Path, video_name: str) -> Path | None:
    """Find matching preprocessed video for a keypoint file."""
    candidates = [
        preprocessed_dir / f"{video_name}.mp4",
        preprocessed_dir / f"{video_name.replace('_2d', '_preprocessed')}.mp4",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def extract_all_frames(video_path: Path) -> list[np.ndarray]:
    """Read all frames from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise IOError(f"Cannot open video: {video_path}")
    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def write_frame_image(frame: np.ndarray, out_path: Path) -> None:
    """Write a single frame to disk as a JPEG image."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)


# ---------------------------------------------------------------------------
# Keypoint / annotation helpers
# ---------------------------------------------------------------------------


def compute_visibility(confidence: float) -> int:
    """Return COCO visibility value from confidence score."""
    if confidence is None or math.isnan(confidence):
        return 0
    if confidence <= 0.0:
        return 0
    if confidence < 0.3:
        return 1
    return 2


def build_keypoints_array(
    frame_keypoints: list[dict[str, Any]], num_keypoints: int = 19
) -> list[float]:
    """Build a flat [x1, y1, v1, ...] array of length num_keypoints * 3."""
    coords: dict[int, tuple[float, float, float]] = {}
    for kp in frame_keypoints:
        kid = int(kp["id"])
        x = float(kp.get("x", 0.0))
        y = float(kp.get("y", 0.0))
        conf = float(kp.get("confidence", 0.0))
        v = compute_visibility(conf)
        coords[kid] = (x, y, float(v))

    flat: list[float] = []
    for kid in range(num_keypoints):
        x, y, v = coords.get(kid, (0.0, 0.0, 0.0))
        flat.extend([x, y, v])
    return flat


def compute_bbox_from_keypoints(keypoints: list[float]) -> list[float]:
    """Compute bounding box [x_min, y_min, w, h] from visible keypoints."""
    xs: list[float] = []
    ys: list[float] = []
    for i in range(0, len(keypoints), 3):
        x, y, v = keypoints[i], keypoints[i + 1], keypoints[i + 2]
        if v > 0:
            xs.append(x)
            ys.append(y)

    if not xs:
        return [0.0, 0.0, 0.0, 0.0]

    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def count_visible_keypoints(keypoints: list[float]) -> int:
    """Count keypoints with visibility > 0."""
    count = 0
    for i in range(0, len(keypoints), 3):
        if keypoints[i + 2] > 0:
            count += 1
    return count


# ---------------------------------------------------------------------------
# COCO generation
# ---------------------------------------------------------------------------


def build_coco_category() -> dict[str, Any]:
    """Build the single category entry for baseball pose."""
    return {
        "id": 1,
        "name": "person",
        "supercategory": "person",
        "keypoints": COCO_KEYPOINT_NAMES,
        "skeleton": COCO_SKELETON,
    }


def process_split(
    cleaned_dir: Path,
    preprocessed_dir: Path,
    images_out_dir: Path,
    split_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Process all cleaned JSONs for a split and return COCO images + annotations."""
    if not cleaned_dir.exists():
        return [], []

    json_files = sorted(p for p in cleaned_dir.rglob("*.json") if p.is_file())

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    image_id_counter = 0
    annotation_id_counter = 0

    for json_path in json_files:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        video_name = data.get("video_name", json_path.stem)
        resolution = data.get("resolution", [1920, 1080])
        width, height = int(resolution[0]), int(resolution[1])
        frames = data.get("frames", [])

        video_path = find_preprocessed_video(preprocessed_dir, video_name)
        if video_path is None:
            continue

        video_frames = extract_all_frames(video_path)

        for frame in frames:
            frame_id = int(frame.get("frame_id", 0))
            if frame_id >= len(video_frames):
                continue

            image_id_counter += 1
            image_name = f"{video_name}_{frame_id:06d}.jpg"
            image_path = images_out_dir / image_name
            write_frame_image(video_frames[frame_id], image_path)

            images.append(
                {
                    "id": image_id_counter,
                    "file_name": image_name,
                    "width": width,
                    "height": height,
                }
            )

            kp_flat = build_keypoints_array(frame.get("keypoints", []))
            bbox = compute_bbox_from_keypoints(kp_flat)
            num_visible = count_visible_keypoints(kp_flat)
            area = bbox[2] * bbox[3]

            annotation_id_counter += 1
            annotations.append(
                {
                    "id": annotation_id_counter,
                    "image_id": image_id_counter,
                    "category_id": 1,
                    "bbox": bbox,
                    "keypoints": kp_flat,
                    "num_keypoints": num_visible,
                    "area": area,
                    "iscrowd": 0,
                }
            )

    return images, annotations


def write_coco_json(
    images: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    out_path: Path,
) -> None:
    """Write COCO-format JSON to disk."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [build_coco_category()],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_coco_json(annotation_path: Path) -> bool:
    """Verify COCO JSON schema and annotation shapes."""
    if not annotation_path.is_file():
        return False

    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for key in ("images", "annotations", "categories"):
        if key not in data:
            return False

    for ann in data.get("annotations", []):
        keypoints = ann.get("keypoints", [])
        if len(keypoints) != 57:
            return False
        bbox = ann.get("bbox", [])
        if len(bbox) != 4:
            return False

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare MMPose COCO-format dataset from cleaned 2D keypoints"
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("data"), help="Root data directory"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify generated COCO annotations and exit",
    )
    args = parser.parse_args(argv)

    paths = resolve_paths(args.data_root)
    paths["logs"].mkdir(parents=True, exist_ok=True)

    if args.verify:
        train_ok = verify_coco_json(paths["annotations"] / "train.json")
        val_ok = verify_coco_json(paths["annotations"] / "val.json")
        if train_ok and val_ok:
            print("Verification passed")
            return 0
        print("Verification failed")
        return 1

    train_images, train_annotations = process_split(
        paths["train_cleaned"],
        paths["preprocessed"],
        paths["images_train"],
        "train",
    )
    val_images, val_annotations = process_split(
        paths["val_cleaned"],
        paths["preprocessed"],
        paths["images_val"],
        "val",
    )

    write_coco_json(
        train_images, train_annotations, paths["annotations"] / "train.json"
    )
    write_coco_json(val_images, val_annotations, paths["annotations"] / "val.json")

    log_path = paths["logs"] / "dataset_prep.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"train_images: {len(train_images)}\n")
        f.write(f"train_annotations: {len(train_annotations)}\n")
        f.write(f"val_images: {len(val_images)}\n")
        f.write(f"val_annotations: {len(val_annotations)}\n")

    print(
        f"Dataset preparation complete: "
        f"train={len(train_images)} images/{len(train_annotations)} annotations, "
        f"val={len(val_images)} images/{len(val_annotations)} annotations"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
