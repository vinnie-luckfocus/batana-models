---
name: 01-2d-pose-recognition-and-cleaning
status: backlog
created: 2026-04-13T00:50:16Z
updated: 2026-04-13T00:50:16Z
progress: 0%
prd: ccpm/prds/01-2d-pose-recognition-and-cleaning.md
github: https://github.com/vinnie-luckfocus/batana-models/issues/1
---

# Epic: 01-2d-pose-recognition-and-cleaning

## Overview

This epic establishes the foundational data pipeline for the batana-models project. The goal is to process 90 raw baseball/softball swing videos through a fully automated pipeline that performs preprocessing, quality control, 2D pose inference via RTMPose-m, trajectory anomaly detection, auto-correction, and mandatory side-by-side visualization generation. The output is a clean, validated 2D keypoint dataset used for downstream fine-tuning and 3D lifting.

## Architecture Decisions

- **Modular Pipeline Design**: The pipeline is split into discrete stages (preprocessing, inference, anomaly detection, correction, visualization) connected via well-defined JSON/CSV interfaces. This ensures each stage can be rerun independently during debugging.
- **Immutable Raw Data**: `data/raw_videos/` is treated as read-only. All intermediate artifacts are versioned via deterministic filenames and written to separate directories (`data/train_2d_keypoints/`, `data/train_2d_cleaned/`, `visuals/2d/`).
- **Locked 2D Model**: RTMPose-m is the primary 2D backbone. A single profiling step on 1000 frames determines if a fallback to YOLOv8-Pose is necessary; otherwise, RTMPose-m is used for the entire dataset.
- **Deterministic Quality Gates**: All acceptance thresholds (`fps >= 24`, `resolution >= 720p`, `TC >= 0.82`) and anomaly flags (`LOW_FPS`, `LOW_RES`, `NO_SWING`, `UNREFINABLE`) are algorithmic and preset to guarantee zero-human-intervention operation.
- **Hybrid Trajectory Refinement**: Small gaps (single frame) are corrected via cubic spline interpolation. Medium gaps (2-5 frames) use a Kalman RTS Smoother. Large gaps (>5 frames) fallback to optical flow tracking combined with a rigid bat geometry constraint. A global TV-L2 pass smooths second-derivative energy.

## Technical Approach

### Python Packages & Dependencies

Core packages locked in `requirements.txt`:
- `torch>=2.0,<2.2` and `torchvision` — inference backend for RTMPose-m.
- `mmpose==1.3.x`, `mmcv-lite`, `mmdet` — RTMPose model loading and execution.
- `opencv-python-headless==4.8.x` — video I/O, preprocessing, optical flow, and visualization rendering.
- `numpy==1.24.x`, `scipy==1.11.x` — numerical operations, cubic spline interpolation (`CubicSpline`), and Kalman smoothing.
- `scikit-image` — histogram equalization and color normalization.
- `pandas` — `2d_cleaning_log.csv` generation and manipulation.
- `tqdm` — progress reporting during batch processing.
- `pyyaml` — configuration parsing under `configs/`.

Model artifacts:
- `models/2d_pose/rtmpose_m_coco.pth` — downloaded from the official MMPose model zoo with SHA-256 verification.

### Data Pipeline

1. **Ingestion**:
   - Scan `data/raw_videos/train/` (75 files) and `data/raw_videos/val/` (15 files) for `.mp4` and `.mov`.
2. **Normalization**:
   - Resample to exactly 30 fps using frame blending/dropping.
   - Scale short side to 1080 and center-crop to 1920x1080.
   - Apply CLAHE histogram equalization and per-channel mean/std color normalization.
3. **Swing Segmentation**:
   - Detect "load", "contact", and "follow-through" using a lightweight heuristic (e.g., wrist velocity peaks + bat orientation change).
   - Extract the core swing and temporally pad/trim to 120 frames. Short clips are symmetrically padded; long clips are centered and trimmed.
4. **Quality Control**:
   - Mark `UNREFINABLE` if original fps < 24 (`LOW_FPS`), resolution < 720p (`LOW_RES`), or swing completeness is not detected (`NO_SWING`).
   - Emit a per-video QC JSON to `logs/qc_report.json`.
   - If usable training samples < 50, trigger an emergency augmentation flag logged to `logs/emergency_augmentation.flag`.

### Pose Inference Pipeline

- **Input**: Preprocessed 1920x1080 @ 30 fps MP4s.
- **Model**: RTMPose-m configured for COCO-17 + 2 custom bat keypoints (`coco-17-plus-bat` schema, ids 0-16 for COCO, 17 for `bat_knob`, 18 for `bat_barrel`).
- **Output Schema** (per frame):
  ```json
  {
    "video_id": "string",
    "frame": 0,
    "keypoints": [
      {"id": 0, "name": "nose", "x": 512.3, "y": 234.1, "score": 0.98},
      ...
      {"id": 17, "name": "bat_knob", "x": 890.0, "y": 450.0, "score": 0.95},
      {"id": 18, "name": "bat_barrel", "x": 920.0, "y": 430.0, "score": 0.94}
    ]
  }
  ```
- **Latency Check**: Profile 1000 frames. If average latency > 35 ms, switch to YOLOv8-Pose as a fallback; abort if neither model achieves < 50 ms.
- **Outputs**: `data/train_2d_keypoints/{video_id}.json` and `data/val_2d_keypoints/{video_id}.json`.

### Trajectory Refinement Pipeline

- **Anomaly Detection**:
  1. Compute per-keypoint Euclidean displacement `d_t = ||P_t - P_{t-1}||` across all frames.
  2. Compute local mu and sigma over a sliding window (e.g., 15 frames); flag jump frames where `d_t > mu + 3*sigma`.
  3. For wrists (id=9, 10), `bat_knob` (id=17), and `bat_barrel` (id=18), compute dense optical flow displacement between consecutive frames. Flag if the predicted vs RTMPose-predicted deviation exceeds 20 px.
  4. Fallback二次判定: if local statistics are unstable (e.g., sigma == 0 or window too short), flag based on acceleration threshold `||P_t - 2P_{t-1} + P_{t-2}|| > 40 px`.
- **Auto-Correction**:
  1. **1-frame gaps**: `scipy.interpolate.CubicSpline` using 5 neighboring frames on each side.
  2. **2-5 frame gaps**: Kalman Smoother (RTS) using a constant-velocity state-space model per keypoint.
  3. **>5 frame gaps**: Lukas-Kanade optical flow tracking for wrists and bat tips, fused with a rigid bat model enforcing bat length within +/- 3% of the median detected bat length.
  4. **Global pass**: temporal total variation (TV-L2) minimization penalizing second-derivative energy `sum_t ||P_{t+1} - 2P_t + P_{t-1}||^2`.
- **Quality Gate**:
  - Compute Trajectory Coherence (TC) after correction as the mean Pearson correlation between adjacent-frame displacements and a smoothed reference.
  - Iterate correction up to 3 times if TC < 0.82.
  - If still below threshold, mark `UNREFINABLE` and exclude from cleaned dataset.
- **Logging**:
  - Append every correction event to `logs/2d_cleaning_log.csv` with columns: `video_id`, `frame`, `keypoint_id`, `anomaly_type`, `correction_method`, `pre_correction_x`, `pre_correction_y`, `post_correction_x`, `post_correction_y`, `tc_score`.

### Visualization Pipeline

- For every usable and corrected video, render a 1920x1080 @ 30 fps side-by-side comparison:
  - **Left pane**: original preprocessed video overlaid with the raw RTMPose skeleton in red.
  - **Right pane**: same video overlaid with the corrected skeleton in green; corrected keypoints are highlighted in yellow.
  - Top overlay: filename and current frame number (`{video_id} | Frame: {N}/120`).
- Skeleton drawing follows the standard COCO skeleton plus a line connecting `bat_knob` (17) to `bat_barrel` (18).
- Output: `visuals/2d/train/{video_id}_2d_compare.mp4` and `visuals/2d/val/{video_id}_2d_compare.mp4`.

## Implementation Strategy

1. **Setup environment and download RTMPose-m weights**
   - Create directory tree, write `requirements.txt`, install dependencies, and download/verify model weights.
2. **Build preprocessing pipeline with quality control**
   - Implement video normalization, swing completeness detection, and QC flagging.
3. **Implement 2D inference and JSON output**
   - Integrate MMPose RTMPose-m, define `coco-17-plus-bat` output schema, and run batch inference.
4. **Build anomaly detection and auto-correction modules**
   - Implement displacement statistics, optical flow consistency checks, fallback acceleration threshold, and the tiered correction strategy.
5. **Generate 2D comparison visualization videos**
   - Build OpenCV-based renderer for side-by-side raw vs corrected skeleton videos.
6. **Validate all outputs and metrics**
   - Run the full pipeline end-to-end, verify latency < 50 ms, usable count >= 70, TC >= 0.82, and presence of all visualization videos and correction logs.

## Task Breakdown Preview

High-level task categories:
- [ ] Environment setup and model weight download
- [ ] Video preprocessing and quality control
- [ ] 2D Pose inference pipeline
- [ ] Trajectory anomaly detection
- [ ] Trajectory auto-correction
- [ ] 2D visualization video generation
- [ ] Validation and logging

## Dependencies

- RTMPose-m pretrained weights (MMPose)
- OpenCV, NumPy, SciPy for preprocessing and visualization
- PyTorch for inference

## Success Criteria (Technical)

- All 90 videos processed; usable count >= 70
- Inference latency < 50 ms/frame on RTX 4090
- Cleaned trajectories pass TC >= 0.82
- Every usable video has a 2D visualization video
- Correction log exists in `logs/2d_cleaning_log.csv`

## Estimated Effort

- 2 weeks (10 working days)
