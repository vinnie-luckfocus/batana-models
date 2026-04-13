import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

# Import the module under test by adding scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.prepare_mmpose_dataset import (
    build_coco_category,
    compute_bbox_from_keypoints,
    compute_visibility,
    count_visible_keypoints,
    find_preprocessed_video,
    verify_coco_json,
)


def _make_cleaned_json(video_name: str, frame_ids: list[int]) -> dict:
    """Build a minimal cleaned keypoint JSON structure."""
    frames = []
    for fid in frame_ids:
        keypoints = []
        for kid in range(19):
            keypoints.append(
                {
                    "id": kid,
                    "x": float(10 + kid * 2),
                    "y": float(10 + kid * 2),
                    "confidence": 0.9 if kid < 17 else 0.5,
                }
            )
        frames.append({"frame_id": fid, "keypoints": keypoints})
    return {
        "video_name": video_name,
        "fps": 30,
        "resolution": [1920, 1080],
        "keypoint_schema": "coco-17-plus-bat",
        "frames": frames,
    }


def _write_dummy_video(path: Path, num_frames: int = 5, size: tuple = (10, 10)) -> None:
    """Write a small grayscale dummy MP4 using OpenCV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, size)
    for _ in range(num_frames):
        frame = np.full((size[1], size[0], 3), 128, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class TestUnitHelpers:
    def test_compute_visibility_high_confidence(self) -> None:
        assert compute_visibility(0.9) == 2

    def test_compute_visibility_low_confidence(self) -> None:
        assert compute_visibility(0.15) == 1

    def test_compute_visibility_zero(self) -> None:
        assert compute_visibility(0.0) == 0

    def test_compute_visibility_nan(self) -> None:
        assert compute_visibility(float("nan")) == 0

    def test_compute_bbox_from_keypoints(self) -> None:
        kps = [
            10.0,
            20.0,
            2.0,
            30.0,
            40.0,
            2.0,
        ] + [0.0, 0.0, 0.0] * 17
        bbox = compute_bbox_from_keypoints(kps)
        assert bbox == [10.0, 20.0, 20.0, 20.0]

    def test_compute_bbox_no_visible(self) -> None:
        kps = [0.0, 0.0, 0.0] * 19
        bbox = compute_bbox_from_keypoints(kps)
        assert bbox == [0.0, 0.0, 0.0, 0.0]

    def test_count_visible_keypoints(self) -> None:
        kps = [0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0] + [0.0, 0.0, 0.0] * 16
        assert count_visible_keypoints(kps) == 2

    def test_find_preprocessed_video_direct_match(self, tmp_path: Path) -> None:
        preprocessed = tmp_path / "preprocessed"
        preprocessed.mkdir()
        video = preprocessed / "swing_001.mp4"
        video.write_text("dummy")
        assert find_preprocessed_video(preprocessed, "swing_001") == video

    def test_find_preprocessed_video_fallback(self, tmp_path: Path) -> None:
        preprocessed = tmp_path / "preprocessed"
        preprocessed.mkdir()
        video = preprocessed / "swing_001_preprocessed.mp4"
        video.write_text("dummy")
        assert find_preprocessed_video(preprocessed, "swing_001_2d") == video

    def test_find_preprocessed_video_missing(self, tmp_path: Path) -> None:
        preprocessed = tmp_path / "preprocessed"
        preprocessed.mkdir()
        assert find_preprocessed_video(preprocessed, "swing_001") is None

    def test_build_coco_category(self) -> None:
        cat = build_coco_category()
        assert cat["id"] == 1
        assert cat["name"] == "person"
        assert len(cat["keypoints"]) == 19
        assert "bat_knob" in cat["keypoints"]
        assert "bat_barrel" in cat["keypoints"]


class TestIntegration:
    def test_coco_schema_valid(self, tmp_path: Path) -> None:
        data_root = tmp_path / "data"
        train_cleaned = data_root / "train_2d_cleaned"
        preprocessed = data_root / "preprocessed"
        train_cleaned.mkdir(parents=True)
        preprocessed.mkdir(parents=True)

        video_name = "test_swing_2d"
        json_path = train_cleaned / "test_swing.json"
        json_path.write_text(json.dumps(_make_cleaned_json(video_name, [0, 1])))

        video_path = preprocessed / f"{video_name}.mp4"
        _write_dummy_video(video_path, num_frames=5, size=(10, 10))

        script = Path(__file__).parent.parent / "scripts" / "prepare_mmpose_dataset.py"
        result = subprocess.run(
            [sys.executable, str(script), "--data-root", str(data_root)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

        coco_path = data_root / "mmpose_baseball" / "annotations" / "train.json"
        assert coco_path.is_file()

        with open(coco_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        assert "images" in coco
        assert "annotations" in coco
        assert "categories" in coco
        assert len(coco["images"]) == 2
        assert len(coco["annotations"]) == 2

        cat = coco["categories"][0]
        assert cat["keypoints"] == [
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

        for ann in coco["annotations"]:
            assert len(ann["keypoints"]) == 57
            assert len(ann["bbox"]) == 4
            assert ann["category_id"] == 1
            assert ann["iscrowd"] == 0
            assert ann["num_keypoints"] == 19
            assert ann["area"] == ann["bbox"][2] * ann["bbox"][3]

        # Check generated images
        for img_entry in coco["images"]:
            img_path = data_root / "mmpose_baseball" / "images" / "train" / img_entry["file_name"]
            assert img_path.is_file()

        log_path = data_root / "logs" / "dataset_prep.log"
        assert log_path.is_file()
        log_text = log_path.read_text(encoding="utf-8")
        assert "train_images: 2" in log_text
        assert "train_annotations: 2" in log_text

    def test_verify_passes_valid_json(self, tmp_path: Path) -> None:
        data_root = tmp_path / "data"
        annotations = data_root / "mmpose_baseball" / "annotations"
        annotations.mkdir(parents=True)

        valid_coco = {
            "images": [{"id": 1, "file_name": "a.jpg", "width": 10, "height": 10}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [0.0, 0.0, 1.0, 1.0],
                    "keypoints": [0.0, 0.0, 2.0] * 19,
                    "num_keypoints": 19,
                    "area": 1.0,
                    "iscrowd": 0,
                }
            ],
            "categories": [build_coco_category()],
        }
        for split in ("train", "val"):
            (annotations / f"{split}.json").write_text(json.dumps(valid_coco))

        script = Path(__file__).parent.parent / "scripts" / "prepare_mmpose_dataset.py"
        result = subprocess.run(
            [sys.executable, str(script), "--data-root", str(data_root), "--verify"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    def test_verify_fails_invalid_json(self, tmp_path: Path) -> None:
        data_root = tmp_path / "data"
        annotations = data_root / "mmpose_baseball" / "annotations"
        annotations.mkdir(parents=True)

        invalid_coco = {
            "images": [],
            "categories": [build_coco_category()],
            # Missing "annotations"
        }
        for split in ("train", "val"):
            (annotations / f"{split}.json").write_text(json.dumps(invalid_coco))

        script = Path(__file__).parent.parent / "scripts" / "prepare_mmpose_dataset.py"
        result = subprocess.run(
            [sys.executable, str(script), "--data-root", str(data_root), "--verify"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1


class TestVerifyHelpers:
    def test_verify_coco_json_valid(self, tmp_path: Path) -> None:
        path = tmp_path / "valid.json"
        path.write_text(
            json.dumps(
                {
                    "images": [],
                    "annotations": [
                        {
                            "id": 1,
                            "image_id": 1,
                            "category_id": 1,
                            "bbox": [0.0, 0.0, 1.0, 1.0],
                            "keypoints": [0.0, 0.0, 2.0] * 19,
                            "num_keypoints": 19,
                            "area": 1.0,
                            "iscrowd": 0,
                        }
                    ],
                    "categories": [build_coco_category()],
                }
            )
        )
        assert verify_coco_json(path) is True

    def test_verify_coco_json_missing_key(self, tmp_path: Path) -> None:
        path = tmp_path / "missing.json"
        path.write_text(json.dumps({"images": [], "annotations": []}))
        assert verify_coco_json(path) is False

    def test_verify_coco_json_wrong_keypoint_length(self, tmp_path: Path) -> None:
        path = tmp_path / "wrong_kp.json"
        path.write_text(
            json.dumps(
                {
                    "images": [],
                    "annotations": [
                        {
                            "id": 1,
                            "image_id": 1,
                            "category_id": 1,
                            "bbox": [0.0, 0.0, 1.0, 1.0],
                            "keypoints": [0.0, 0.0, 2.0] * 10,
                        }
                    ],
                    "categories": [build_coco_category()],
                }
            )
        )
        assert verify_coco_json(path) is False

    def test_verify_coco_json_wrong_bbox_length(self, tmp_path: Path) -> None:
        path = tmp_path / "wrong_bbox.json"
        path.write_text(
            json.dumps(
                {
                    "images": [],
                    "annotations": [
                        {
                            "id": 1,
                            "image_id": 1,
                            "category_id": 1,
                            "bbox": [0.0, 0.0],
                            "keypoints": [0.0, 0.0, 2.0] * 19,
                        }
                    ],
                    "categories": [build_coco_category()],
                }
            )
        )
        assert verify_coco_json(path) is False
