---
name: 01-2d-pose-recognition-and-cleaning
description: 2D Pose recognition, data preprocessing, quality control, anomaly detection, auto-correction, and 2D visualization video generation
status: backlog
created: 2026-04-13T00:50:16Z
---

# 01 - 2D Pose Recognition and Cleaning

## Executive Summary

Establish the project foundation by running RTMPose-m 2D inference on raw videos, preprocessing the data, enforcing quality control, performing 2D trajectory anomaly detection and auto-correction, and generating mandatory 2D visualization videos. This phase produces a clean 2D dataset used for downstream fine-tuning and 3D lifting.

## Problem Statement

Raw videos vary in resolution, frame rate, lighting, and camera angle. Direct 2D pose inference on unprocessed footage produces noisy trajectories with jump artifacts, especially during high-speed bat swings. Without automated cleaning and visualization, low-quality data would propagate into model training and degrade final accuracy.

## Requirements

### Functional Requirements

1. **Project Setup & Environment**
   - Initialize the repository structure under `data/`, `models/`, `configs/`, `scripts/`, `logs/`, `reports/`, and `visuals/`.
   - Install and lock dependency versions in `requirements.txt`.
   - Download RTMPose-m COCO pretrained weights to `models/2d_pose/rtmpose_m_coco.pth` and verify SHA-256.

2. **Data Preprocessing**
   - Accept raw MP4/MOV files from `data/raw_videos/train/` (75 files) and `data/raw_videos/val/` (15 files).
   - Auto-normalize each video:
     - Resample to 30 fps.
     - Scale/crop to 1920x1080.
     - Extract the core swing segment and pad/trim to exactly 120 frames (4.0 seconds).
     - Apply histogram equalization and color normalization.
   - Mark files as `UNREFINABLE` if original fps < 24 (`LOW_FPS`) or resolution < 720p (`LOW_RES`).
   - Run Swing Completeness Detector; mark as `UNREFINABLE` (`NO_SWING`) if load -> contact -> follow-through is not detected.
   - Target usable video count >= 70 out of 90 (>= 78%). If usable training videos fall below 50, trigger emergency data augmentation mode.

3. **2D Pose Inference**
   - Run RTMPose-m inference on all usable videos.
   - Keypoint schema: extended COCO-17 + 2 bat keypoints = 19 keypoints (`coco-17-plus-bat`).
   - Output per-video JSON to `data/train_2d_keypoints/` and `data/val_2d_keypoints/`.

4. **2D Trajectory Anomaly Detection**
   - Compute per-keypoint Euclidean displacement `d_t = ||P_t - P_{t-1}||`.
   - Flag jump frames where `d_t > mu + 3*sigma`.
   - For wrists (id=9, 10), `bat_knob` (id=17), and `bat_barrel` (id=18), add optical-flow consistency check; flag if predicted vs actual deviation > 20 px.
   - Fallback二次判定: if local statistics fail, use acceleration threshold `||P_t - 2P_{t-1} + P_{t-2}|| > 40 px`.

5. **2D Trajectory Auto-Correction**
   - Single isolated frames: cubic spline interpolation.
   - 2-5 consecutive jump frames: Kalman Smoother (RTS).
   - >5 consecutive frames: optical flow tracking + rigid bat model constraint (bat length tolerance +/- 3%).
   - Global pass: temporal total variation minimization for second-derivative energy minimization.
   - Post-correction quality gate: TC >= 0.82. Agent may iterate up to 3 times; if still below threshold, mark video `UNREFINABLE` and exclude from training.
   - Log all corrections to `logs/2d_cleaning_log.csv`.
   - Write cleaned outputs to `data/train_2d_cleaned/` and `data/val_2d_cleaned/`.

6. **2D Visualization Video Generation (Mandatory Deliverable)**
   - For every corrected video, render a side-by-side comparison video:
     - Left: original video + raw 2D skeleton (red).
     - Right: original video + corrected 2D skeleton (green).
     - Highlight corrected keypoints in yellow on the right.
     - Overlay filename and frame number at top.
   - Resolution 1920x1080, 30 fps, 120 frames.
   - Output to `visuals/2d/train/` and `visuals/2d/val/`.

### Non-Functional Requirements

- Single-frame 2D inference latency < 50 ms on RTX 4090.
- All processing must be fully automated with zero human intervention.
- Raw videos stored read-only; intermediate outputs versioned and reproducible.

## Success Criteria

- All 90 videos are processed and flagged appropriately.
- Usable video count >= 70; cleaned trajectories are physically plausible.
- Every usable video has a corresponding 2D visualization video in `visuals/2d/`.
- `logs/2d_cleaning_log.csv` exists and documents all corrections.

## Constraints & Assumptions

- RTMPose-m is the locked 2D model. Fallback to YOLOv8-Pose only if average inference over 1000 frames exceeds 35 ms.
- Videos contain real baseball/softball athletes performing a full swing.
- Agent operates with zero human intervention; all decisions are algorithmic and deterministic based on preset thresholds.

## Dependencies

None. This is the first phase.

## Out of Scope

- 2D model fine-tuning (covered in PRD 02).
- 3D lifting, optimization, or visualization (covered in PRDs 03-04).
- End-to-end pipeline integration and playback tools (covered in PRD 05).
