#!/usr/bin/env python3
"""2D pose inference using RTMPose-m on preprocessed videos.

Outputs per-video JSON with keypoints following the schema defined in
configs/keypoint_schema.yaml. Bat keypoints are heuristically placed
relative to wrists when the base model only produces COCO-17 keypoints.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

# MMPose imports with fallback
try:
    from mmpose.apis import init_model, inference_topdown
    from mmpose.structures import merge_data_samples
    HAS_MMPOSE = True
except Exception as exc:
    print(f"WARNING: Could not import mmpose: {exc}")
    HAS_MMPOSE = False

import yaml

# ---------------------------------------------------------------------------
# Paths / schema
# ---------------------------------------------------------------------------

def load_keypoint_schema(schema_path: Path) -> dict[str, Any]:
    """Load keypoint schema from YAML."""
    with open(schema_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_paths(data_root: Path) -> dict[str, Path]:
    """Return canonical paths for inference I/O."""
    return {
        "preprocessed": data_root / "preprocessed",
        "train_out": data_root / "train_2d_keypoints",
        "val_out": data_root / "val_2d_keypoints",
        "schema": Path("configs") / "keypoint_schema.yaml",
        "model_dir": Path("models") / "2d_pose",
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_pose_model(model_dir: Path, device: str = "cpu") -> Any:
    """Initialize RTMPose-m model. Returns None if MMPose unavailable."""
    config_path = Path("configs") / "rtmpose_m_256x192.py"
    checkpoint_path = model_dir / "rtmpose_m_coco.pth"

    if not HAS_MMPOSE:
        return None
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # If config doesn't exist, use a minimal inline config
    if not config_path.exists():
        config_path = _write_minimal_rtmpose_config(config_path)

    model = init_model(str(config_path), str(checkpoint_path), device=device)
    return model


def _write_minimal_rtmpose_config(config_path: Path) -> Path:
    """Write a minimal RTMPose-m config for COCO 17 keypoints."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    content = """
# Minimal RTMPose-m config for top-down COCO 17 inference
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.67,
        widen_factor=0.75,
        channel_attention=True,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmpose/v1/projects/'
            'rtmpose/cspnext-m_udp-aic-coco_210e-256x192-f2f7d6f2_20230126.pth')),
    head=dict(
        type='RTMCCHead',
        in_channels=768,
        out_channels=17,
        input_size=(192, 256),
        in_featuremap_size=(6, 8),
        simcc_split_ratio=2.0,
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.0,
            drop_path=0.0,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True),
        decoder=dict(
            type='SimCCLabel',
            input_size=(192, 256),
            smoothing_type='gaussian',
            sigma=(4.9, 5.66),
            simcc_split_ratio=2.0,
            label_smooth_weight=0.0,
            use_dark=False)),
    test_cfg=dict(flip_test=True))
"""
    config_path.write_text(content.strip() + "\n", encoding="utf-8")
    return config_path


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def run_inference_on_video(model: Any, video_path: Path) -> list[dict[str, Any]]:
    """Run 2D pose inference on each frame of a preprocessed video."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if model is None:
        # Fallback: return dummy keypoints when model unavailable
        return _dummy_keypoints_for_frames(frames, fps, width, height)

    results = []
    for idx, frame in enumerate(frames):
        result = inference_topdown(model, frame)
        data_sample = merge_data_samples(result)
        pred_instances = data_sample.get("pred_instances", {})
        keypoints = pred_instances.get("keypoints", np.array([]))
        scores = pred_instances.get("keypoint_scores", np.array([]))

        # Take highest-scoring person if multiple
        if keypoints.ndim == 3 and len(keypoints) > 0:
            person_scores = scores.mean(axis=1) if scores.ndim == 2 else scores
            best_idx = int(np.argmax(person_scores))
            kpts = keypoints[best_idx]
            confs = scores[best_idx] if scores.ndim == 2 else np.full(len(kpts), 0.5)
        elif keypoints.ndim == 2:
            kpts = keypoints
            confs = scores if scores.ndim == 1 else np.full(len(kpts), 0.5)
        else:
            kpts = np.zeros((17, 2))
            confs = np.zeros(17)

        frame_result = build_frame_result(idx, kpts, confs, width, height)
        results.append(frame_result)

    return results


def _dummy_keypoints_for_frames(
    frames: list[np.ndarray], fps: float, width: int, height: int
) -> list[dict[str, Any]]:
    """Generate dummy keypoints for testing when model is unavailable."""
    results = []
    for idx in range(len(frames)):
        kpts = np.tile(np.array([[width * 0.5, height * 0.5]]), (17, 1))
        confs = np.full(17, 0.5)
        results.append(build_frame_result(idx, kpts, confs, width, height))
    return results


def build_frame_result(
    frame_id: int,
    keypoints: np.ndarray,
    scores: np.ndarray,
    img_width: int,
    img_height: int,
) -> dict[str, Any]:
    """Build a single frame result dict and append bat keypoints heuristically."""
    kpts_list: list[dict[str, Any]] = []

    # COCO-17 keypoints
    for i in range(len(keypoints)):
        x, y = float(keypoints[i][0]), float(keypoints[i][1])
        # Clamp to image bounds
        x = max(0.0, min(x, float(img_width)))
        y = max(0.0, min(y, float(img_height)))
        kpts_list.append({"id": i, "x": x, "y": y, "confidence": float(scores[i])})

    # Heuristic bat keypoints (id 17, 18) placed relative to wrists (9=left, 10=right)
    # Handedness agnostic: place bat along the vector between wrists, extending outward
    if len(keypoints) >= 11:
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]
        vec = right_wrist - left_wrist
        length = np.linalg.norm(vec) + 1e-6
        unit = vec / length
        perp = np.array([-unit[1], unit[0]])

        # Bat knob closer to trailing wrist (choose midpoint shifted slightly)
        bat_knob = (left_wrist + right_wrist) * 0.5 - unit * (length * 0.3) + perp * (length * 0.1)
        bat_barrel = bat_knob + unit * (length * 1.8)

        for kid, pt in [(17, bat_knob), (18, bat_barrel)]:
            x = max(0.0, min(float(pt[0]), float(img_width)))
            y = max(0.0, min(float(pt[1]), float(img_height)))
            kpts_list.append({"id": kid, "x": x, "y": y, "confidence": 0.5})
    else:
        # Fallback: dummy bat keypoints at image center
        for kid in [17, 18]:
            kpts_list.append(
                {
                    "id": kid,
                    "x": float(img_width) * 0.5,
                    "y": float(img_height) * 0.5,
                    "confidence": 0.0,
                }
            )

    return {"frame_id": frame_id, "keypoints": kpts_list}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_video_keypoints(
    output_path: Path,
    video_name: str,
    fps: float,
    resolution: list[int],
    schema_name: str,
    frames: list[dict[str, Any]],
) -> None:
    """Write per-video keypoint JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "video_name": video_name,
        "fps": fps,
        "resolution": resolution,
        "keypoint_schema": schema_name,
        "frames": frames,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run 2D pose inference on preprocessed videos")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args(argv)

    paths = resolve_paths(args.data_root)
    schema = load_keypoint_schema(paths["schema"])
    schema_name = schema.get("schema_name", "coco-17-plus-bat")

    model = load_pose_model(paths["model_dir"], device=args.device)
    if model is None:
        print("WARNING: MMPose not available; generating dummy keypoints for structure testing.")

    video_files = sorted(paths["preprocessed"].rglob("*_preprocessed.mp4"))
    if not video_files:
        print("No preprocessed videos found.")
        return 0

    for video_path in tqdm(video_files, desc="Running 2D inference"):
        frames_results = run_inference_on_video(model, video_path)
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if "train" in str(video_path).lower() or "train" in video_path.parent.name:
            out_dir = paths["train_out"]
        else:
            out_dir = paths["val_out"]

        out_name = video_path.stem.replace("_preprocessed", "") + "_2d.json"
        out_path = out_dir / out_name
        write_video_keypoints(
            out_path,
            video_name=video_path.stem,
            fps=fps,
            resolution=[width, height],
            schema_name=schema_name,
            frames=frames_results,
        )

    print(f"Inference complete. Output written to {paths['train_out']} and {paths['val_out']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
