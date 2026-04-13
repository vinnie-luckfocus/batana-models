---
name: 04-3d-pose-optimization-and-validation
description: Test-Time Adaptation, 2-layer Transformer post-refinement network, SMPL constraint module, and 3D validation with quantitative targets
status: backlog
created: 2026-04-13T00:50:16Z
---

# 04 - 3D Pose Optimization and Validation

## Executive Summary

Optimize the 3D pipeline through Test-Time Adaptation (TTA), train a mandatory 2-layer Transformer post-refinement network supervised on cleaned 3D trajectories, and integrate a SMPL constraint module. Validate against strict 3D metrics and ensure temporal alignment between 2D and 3D apex frames.

## Problem Statement

Even after SMPL fitting and sliding window correction, residual errors in monocular 3D lifting remain. Without TTA and a learned post-refinement network, the system cannot generalize corrections across new videos. A dedicated optimization and validation phase is needed to lock in physical plausibility and temporal coherence.

## Requirements

### Functional Requirements

1. **Test-Time Adaptation (TTA)**
   - Implement TTA for MotionBERT during inference on each clip.
   - Optimize using bone length loss and temporal smoothness loss.
   - Maximum 10 gradient steps per clip.
   - Save TTA configuration to `models/3d_pose/motionbert_tta_config.yaml`.
   - If GPU OOM occurs during TTA, report error and halt (no silent fallback).

2. **Post-Refinement Network (Mandatory Deliverable)**
   - Architecture: lightweight 2-layer Transformer.
   - Input: raw MotionBERT 3D output sequences.
   - Target: Agent-corrected 3D trajectories from Phase 03.
   - Output: residual correction added to raw lifting output.
   - Train the network with SMPL constraint losses (bone length + joint angle regularization) jointly applied.
   - Save trained weights to `models/3d_pose/refiner_net.pth`.
   - This network is mandatory and must not be skipped regardless of dataset size.

3. **SMPL Constraint Module**
   - Encode bone length and joint angle constraints as differentiable losses.
   - Apply during both TTA optimization and post-refinement network training.
   - Ensure gradients flow back through the refinement network and TTA steps.

4. **3D Validation Testing**
   - Run the full optimized pipeline (TTA + post-refinement network) on the validation set.
   - Compute metrics:
     - 3D Temporal Coherence (3D-TC) = 1 / (1 + mean(||Q_t - 2Q_{t-1} + Q_{t-2}||)) — target > 0.78
     - Bone Length Consistency (BLC) = mean(sigma_bone / mu_bone) * 100% — target < 4%
     - Physical Plausibility (PP) = ratio of frames with unreasonable joint angles — target < 8%
     - Pose Similarity (PS) = average DTW distance between validation poses and a standard swing template — target < 180
   - Apex frame alignment: compute `bat_barrel` (id=18) horizontal displacement speed peak in both 2D and 3D sequences; offset must be < 2 frames.
   - Generate `reports/3d_validation_report.md` with per-video and aggregate results.
   - If validation fails, Agent must auto-adjust correction weights (lambda_2, lambda_3, lambda_4 within locked ranges) and re-run without human intervention.

### Non-Functional Requirements

- TTA + inference must complete within the per-video budget (< 90 seconds target, < 3 minutes hard limit for 120 frames on RTX 4090).
- Training the post-refinement network must fit in 24 GB VRAM.
- All optimization and validation steps are fully automated.

## Success Criteria

- TTA config and post-refinement network weights exist and are versioned.
- Validation report shows 3D-TC > 0.78, BLC < 4%, PP < 8%, PS < 180.
- Apex frame offset between 2D and 3D is < 2 frames for all validation videos.
- If initial validation fails, Agent completes a self-correction loop and revalidates automatically.

## Constraints & Assumptions

- Locked hardware: RTX 4090.
- Post-refinement network is mandatory and cannot be omitted.
- All targets are hard thresholds; Agent is empowered to iterate automatically to meet them.

## Dependencies

- PRD 03 (`03-3d-pose-lifting-and-cleaning`)
  - Corrected 3D trajectories in `data/train_3d_cleaned/` and `data/val_3d_cleaned/`.
  - SMPL fitting and IK modules (`scripts/agent/smpl_fitter.py`).

## Out of Scope

- 2D inference and cleaning (covered in PRDs 01-02).
- End-to-end integration, playback tooling, and final packaging (covered in PRD 05).
