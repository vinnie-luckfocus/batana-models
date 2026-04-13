---
name: 02-2d-pose-fine-tuning-and-validation
status: backlog
created: 2026-04-13T01:53:20Z
updated: 2026-04-13T01:53:20Z
progress: 0%
prd: ccpm/prds/02-2d-pose-fine-tuning-and-validation.md
github:
---

# Epic: 02-2d-pose-fine-tuning-and-validation

## Overview

This epic fine-tunes RTMPose-m for the baseball swing domain using cleaned 2D trajectories from Phase 01 as pseudo-Ground Truth. Training runs with locked hyperparameters and strong augmentation, followed by quantitative validation on a held-out set to ensure the model meets strict performance targets.

## Goal

Produce a domain-adapted 2D pose estimator that achieves Temporal Coherence > 0.82, Keypoint Stability > 92%, and Missing Rate < 5% on validation, with zero manual hyperparameter tuning.

## Acceptance Criteria

- Dataset is prepared from `data/train_2d_cleaned/` into the MMPose training format, preserving the 19-keypoint `coco-17-plus-bat` schema (bat keypoints at indices 17 and 18).
- RTMPose-m is fine-tuned with locked hyperparameters: freeze the first 3 backbone stages; train remaining layers and regression head with batch size 16, AdamW optimizer, initial lr 5e-4, cosine annealing, max epochs 50, early stopping patience 10, and weight decay 1e-4.
- Loss function combines Smooth L1 Loss with a temporal consistency loss (L1 difference between adjacent-frame predictions).
- Strong augmentation is applied: rotation ±30°, perspective ±15%, brightness ±30% / contrast ±20% / saturation ±20%, Gaussian noise (sigma=5), horizontal flip, temporal downsampling to 15 fps, and Mosaic/MixUp if available.
- Best checkpoint is saved to `models/2d_pose/rtmpose_m_finetuned.pth`; training logs are stored under `logs/training_logs/`; and a config snapshot is persisted for reproducibility.
- Validation on the held-out val set achieves TC > 0.82, KS > 92%, and Missing Rate < 5%.
- A validation report is generated at `reports/2d_validation_report.md` with per-video and aggregate statistics.
- If validation fails, the Agent completes at least one autonomous self-correction loop back to Phase 01 cleaning and re-trains without human intervention.

## Scope Boundaries

### In Scope
- Dataset preparation from cleaned 2D trajectories
- Fine-tuning RTMPose-m with locked hyperparameters
- Model checkpoint and log persistence
- Quantitative validation and reporting
- Automated loopback on validation failure

### Out of Scope
- 3D lifting, optimization, or test-time augmentation (TTA)
- Visualization playback tools and end-to-end integration
- CPU fallback deployment configuration

## Architecture / Key Decisions

- **Pseudo-Ground Truth Training**: Phase 01 cleaned trajectories serve as labels to adapt the generic COCO-pretrained model to baseball-specific poses and bat handling.
- **Progressive Unfreezing**: Freezing the first 3 backbone stages retains low-level visual features while allowing later stages and the regression head to learn domain-specific patterns.
- **Locked Hyperparameters**: All training settings are preset and immutable. The only permitted deviation is ReduceLROnPlateau (factor=0.1, patience=5) within the range [1e-5, 1e-3] when validation loss plateaus.
- **Aggressive Augmentation**: Extensive domain-specific augmentation compensates for a limited video corpus and improves model generalization.
- **Autonomous Feedback Loop**: Failed validation automatically triggers a return to Phase 01 refinement strategies, enforcing zero-human-intervention operation.

## Tasks Created

(To be populated during epic planning)
