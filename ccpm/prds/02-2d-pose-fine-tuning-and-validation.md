---
name: 02-2d-pose-fine-tuning-and-validation
description: Prepare dataset from cleaned 2D trajectories, fine-tune RTMPose-m with locked hyperparameters, and validate against quantitative targets
status: backlog
created: 2026-04-13T00:50:16Z
---

# 02 - 2D Pose Fine-Tuning and Validation

## Executive Summary

Use the cleaned 2D trajectories from Phase 01 as pseudo-Ground Truth to fine-tune RTMPose-m for the baseball swing domain. Train with locked hyperparameters and strong augmentation, then validate on the held-out validation set to ensure TC > 0.82, KS > 92%, and Missing Rate < 5%.

## Problem Statement

Off-the-shelf RTMPose-m is trained on generic COCO poses and may underperform on baseball-specific body configurations, bat handling, and extreme swing angles. Fine-tuning on domain-cleaned data is required to improve accuracy and stability without overfitting to a small video corpus.

## Requirements

### Functional Requirements

1. **Dataset Preparation**
   - Convert cleaned 2D outputs from `data/train_2d_cleaned/` into the training format expected by MMPose.
   - Maintain the 19-keypoint schema (`coco-17-plus-bat`) with bat keypoints at indices 17 and 18.
   - Split prepared data into training and internal validation subsets (or use the separate val directory for evaluation).

2. **Fine-Tuning RTMPose-m (Locked Hyperparameters)**
   - Backbone strategy: freeze the first 3 stages; unfreeze remaining layers and regression head.
   - Batch size: 16
   - Optimizer: AdamW
   - Initial learning rate: 5e-4
   - LR scheduler: cosine annealing
   - Max epochs: 50
   - Early stopping patience: 10
   - Weight decay: 1e-4
   - Loss: Smooth L1 Loss + temporal consistency loss (L1 difference between adjacent-frame predictions).
   - Data augmentation:
     - Random rotation +/- 30 degrees
     - Perspective transform +/- 15%
     - Brightness +/- 30%, contrast +/- 20%, saturation +/- 20%
     - Gaussian noise (sigma=5)
     - Horizontal flip
     - Temporal downsampling to simulate 15 fps
     - Mosaic / MixUp if available
   - Agent autonomy: when validation loss plateaus, Agent may invoke ReduceLROnPlateau (factor=0.1, patience=5) within [1e-5, 1e-3] in powers of 10. No other hyperparameters may be changed.

3. **Model Saving**
   - Save best checkpoint to `models/2d_pose/rtmpose_m_finetuned.pth`.
   - Store training logs under `logs/training_logs/`.
   - Persist training config snapshot for reproducibility.

4. **Validation Testing**
   - Run inference on `data/raw_videos/val/` using the fine-tuned model.
   - Compute metrics:
     - Temporal Coherence (TC) = 1 / (1 + mean(||P_t - 2P_{t-1} + P_{t-2}||)) — target > 0.82
     - Keypoint Stability (KS) = 1 - (jump frames / total frames) — target > 92%
     - Missing Rate = proportion of keypoints with confidence < 0.3 — target < 5%
   - Generate `reports/2d_validation_report.md` with per-video and aggregate statistics.
   - If validation fails, Agent automatically returns to Phase 01 refinement strategies (correction algorithm adjustments) without human intervention.

### Non-Functional Requirements

- Training must fit within 24 GB VRAM (RTX 4090).
- Training pipeline must be fully automated; no manual tuning of locked hyperparameters.
- All raw data remains read-only; only model artifacts and logs are written.

## Success Criteria

- Fine-tuned model saved at `models/2d_pose/rtmpose_m_finetuned.pth`.
- Validation report shows TC > 0.82, KS > 92%, and Missing Rate < 5%.
- If metrics are not met, Agent completes at least one self-correction loop back to cleaning and re-trains automatically.

## Constraints & Assumptions

- Hyperparameters are locked except for LR reduction via ReduceLROnPlateau.
- Training set size may be reduced by Phase 01 quality control; emergency augmentation is triggered if usable training videos drop below 50.
- Target hardware is RTX 4090.

## Dependencies

- PRD 01 (`01-2d-pose-recognition-and-cleaning`)
  - Cleaned 2D trajectories in `data/train_2d_cleaned/` and `data/val_2d_cleaned/`.
  - Quality-controlled list of usable training and validation videos.

## Out of Scope

- 3D lifting, optimization, or TTA.
- Visualization playback tools and end-to-end integration.
- CPU fallback deployment configuration.
