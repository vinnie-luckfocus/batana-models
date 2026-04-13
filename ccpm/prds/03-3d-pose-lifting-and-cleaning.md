---
name: 03-3d-pose-lifting-and-cleaning
description: MotionBERT 2D-to-3D lifting, 3D anomaly detection, 3D trajectory auto-correction, and 3D visualization video generation
status: backlog
created: 2026-04-13T00:50:16Z
---

# 03 - 3D Pose Lifting and Cleaning

## Executive Summary

Lift the fine-tuned 2D pose sequences into 3D using MotionBERT, then detect and automatically correct 3D anomalies via SMPL prior fitting, IK refinement, sliding window optimization, and global bundle adjustment. Generate mandatory 3D visualization videos for every corrected sequence.

## Problem Statement

Monocular 3D lifting is inherently ambiguous, especially depth (z-axis) estimation. Diverse camera angles (side, front, oblique) compound the problem. Without strong biomechanical constraints and visualization, 3D trajectories can exhibit impossible bone stretches, joint angles, ground penetration, and root displacement jumps.

## Requirements

### Functional Requirements

1. **2D-to-3D Lifting**
   - Use MotionBERT to lift 2D sequences from Phase 02 into 3D.
   - Input: cleaned 2D trajectories (`data/train_2d_cleaned/`, `data/val_2d_cleaned/`).
   - Output: per-sequence 3D keypoints (19 keypoints, x/y/z in cm) with hip midpoint (COCO-11 and COCO-12 midpoint) as origin.
   - Before lifting, automatically estimate camera viewpoint from 2D detections and write to metadata; apply stronger SMPL depth regularization for non-side views.
   - Save raw 3D outputs to `data/train_3d_keypoints/` and `data/val_3d_keypoints/`.

2. **3D Anomaly Detection**
   - Bone length jump: any frame where a predefined bone pair length deviates > 5% from the sequence median.
   - Unreasonable joint angles: elbow < 30 degrees or > 180 degrees; knee < 10 degrees or > 170 degrees.
   - Root displacement: hip midpoint displacement between adjacent frames > 15 cm.
   - Ground penetration: ankle or knee y-coordinate < 0.

3. **3D Trajectory Auto-Correction**
   - SMPL prior fitting: project 2D detections onto SMPL, optimize with Levenberg-Marquardt to minimize 2D reprojection error while enforcing plausible bone lengths. Use result as initial corrected pose.
   - IK refinement: use Jacobian-based IK for bat-upper limb relationships, preserving `bat_knob` to wrists constraints.
   - Sliding window optimization (window size = 7 frames): minimize bone length variance + joint acceleration + ground penetration penalty across the window.
   - Global bundle adjustment over the full sequence with locked objective:
     - `L_total = lambda_1 * L_reproj + lambda_2 * L_bone_length + lambda_3 * L_temporal_smooth + lambda_4 * L_ground_penetration`
     - Weights: lambda_1 = 1.0 (fixed); lambda_2 = 2.0 ([1.0, 5.0], step 0.5); lambda_3 = 0.5 ([0.1, 2.0], step 0.5); lambda_4 = 2.0 ([1.0, 10.0], step 0.5).
   - Hard constraints:
     - Bone length standard deviation across time < 2% of median bone length.
     - Human bone length temporal variation tolerance = 2.5%; exceeding triggers re-correction.
     - Prioritize z-axis correction due to monocular depth ambiguity.
   - Log all corrections to `logs/3d_cleaning_log.csv`.
   - Save corrected 3D outputs to `data/train_3d_cleaned/` and `data/val_3d_cleaned/`.

4. **3D Visualization Video Generation (Mandatory Deliverable)**
   - For every corrected sequence, render a side-by-side 3D skeleton animation:
     - Left: raw MotionBERT 3D skeleton (red) rotating around hip origin (+/- 30 degrees on Y axis).
     - Right: corrected 3D skeleton (green) with same auto-rotation.
     - Top overlay: filename, frame number, bone length deviation percentage.
     - Yellow highlight on corrected joints/bones on the right.
     - Bottom timeline bar: red segments = original anomaly intervals, green = corrected normal intervals.
   - Resolution 1920x1080, 30 fps, 120 frames.
   - Output to `visuals/3d/train/` and `visuals/3d/val/`.

### Non-Functional Requirements

- Lifting and correction pipeline must run without human intervention.
- GPU memory must stay within 24 GB; if OOM occurs, pipeline stops with an error (no silent downgrade).
- Raw and cleaned 2D inputs remain read-only.

## Success Criteria

- All usable videos have raw and corrected 3D sequences.
- Corrected bone length temporal standard deviation < 2% for all sequences.
- Every usable sequence has a corresponding 3D visualization video in `visuals/3d/`.
- `logs/3d_cleaning_log.csv` exists and documents all corrections.

## Constraints & Assumptions

- MotionBERT is the locked 3D model. Downgrade to VideoPose3D-243f only if usable videos < 30.
- Agent makes all correction decisions autonomously based on preset thresholds.
- Input 2D data is already cleaned and fine-tuned from Phase 02.

## Dependencies

- PRD 02 (`02-2d-pose-fine-tuning-and-validation`)
  - Fine-tuned 2D model (`models/2d_pose/rtmpose_m_finetuned.pth`).
  - Cleaned 2D trajectories for training and validation.

## Out of Scope

- TTA and post-refinement network training (covered in PRD 04).
- Final validation report aggregation beyond 3D cleaning logs.
- End-to-end integration and playback UI (covered in PRD 05).
