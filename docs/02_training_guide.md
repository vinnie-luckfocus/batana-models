# Epic 02: 2D Pose Fine-Tuning and Validation Guide

This guide explains how to run the full fine-tuning pipeline for the baseball-specific 2D pose model.

## Pipeline Overview

The pipeline consists of four stages executed in order:

1. **Dataset Preparation** — convert cleaned 2D keypoints into MMPose COCO-format dataset.
2. **Training** — fine-tune RTMPose-m on the baseball dataset.
3. **Validation** — compute Temporal Coherence (TC), Keypoint Stability (KS), and Missing Rate (MR).
4. **Report Generation** — produce a markdown validation report.

## Step-by-Step Instructions

### 1. Prepare the Dataset

```bash
python scripts/prepare_mmpose_dataset.py --data-root data
```

This reads cleaned keypoint JSONs from `data/train_2d_cleaned/` and `data/val_2d_cleaned/`, extracts frames from preprocessed videos, and writes:

- `data/mmpose_baseball/images/train/`
- `data/mmpose_baseball/images/val/`
- `data/mmpose_baseball/annotations/train.json`
- `data/mmpose_baseball/annotations/val.json`

To verify the generated COCO files without regenerating images:

```bash
python scripts/prepare_mmpose_dataset.py --data-root data --verify
```

### 2. Train the Model

```bash
python scripts/train_2d_pose.py --config configs/rtmpose_m_finetune_baseball.py
```

Training will:

- validate locked hyperparameters against the config,
- snapshot the config to `logs/training_logs/rtmpose_m_finetune_baseball_snapshot.py`,
- write a JSON log to `logs/training_logs/training_log.json`.

If MMPose is unavailable, a dummy training loop runs instead and still produces the artifacts.

### 3. Validate the Model

```bash
python scripts/validate_2d_pose.py \
  --config configs/rtmpose_m_finetune_baseball.py \
  --model models/2d_pose/rtmpose_m_finetuned.pth \
  --val-dir data/mmpose_baseball/images/val \
  --output logs/validation_metrics.json
```

If the checkpoint or MMPose environment is missing, the script falls back to dummy inference and prints a warning. The exit code is non-zero when any metric misses its target:

| Metric | Target |
|--------|--------|
| TC  | \> 0.82 |
| KS  | \> 0.92 |
| MR  | \< 0.05 |

### 4. Generate the Validation Report

```bash
python scripts/generate_validation_report.py \
  --metrics-json logs/validation_metrics.json \
  --output reports/2d_validation_report.md
```

The report includes aggregate pass/fail indicators, a per-video table, and traceability metadata.

### Run All Stages at Once

Use the pipeline orchestrator to execute all four stages with automatic self-correction:

```bash
python scripts/run_epic_02_pipeline.py --data-root data
```

If validation fails, the orchestrator automatically triggers Phase 01 re-cleaning in `--strict-mode` and retries the full pipeline once.

## Locked Hyperparameters

The following hyperparameters are locked and enforced by `scripts/train_2d_pose.py`. Overriding them via the config or CLI will raise an error.

| Parameter | Value | Mutable? |
|-----------|-------|----------|
| `model.backbone.frozen_stages` | 3 | No |
| `optim_wrapper.optimizer.lr` | 5e-4 | No |
| `train_dataloader.batch_size` | 16 | No |
| `optim_wrapper.optimizer.weight_decay` | 1e-4 | No |
| `train_cfg.max_epochs` | 50 | No |

These values were chosen to preserve pretrained COCO features (freeze first 3 stages) while allowing stable fine-tuning on the small baseball dataset.

## Troubleshooting

### Out of Memory (OOM)

- Reduce the number of workers in the config (`train_dataloader.num_workers`).
- Enable mixed-precision training by adding `amp=True` to the runner config if supported by your GPU.
- Use a smaller input resolution only if PRD acceptance criteria still allow it.

### Missing Annotations

- Ensure `data/train_2d_cleaned/` and `data/val_2d_cleaned/` contain the cleaned keypoint JSONs from Epic 01.
- Check that preprocessed videos exist in `data/preprocessed/` with matching filenames.
- Run dataset prep with `--verify` to validate COCO JSON structure.

### Strict-Mode Re-Cleaning Loop

If validation fails (TC ≤ 0.82, KS ≤ 0.92, or MR ≥ 0.05), the pipeline orchestrator will:

1. Run `scripts/detect_anomalies_2d.py`.
2. Run `scripts/correct_trajectories_2d.py --strict-mode` with tighter thresholds.
3. Re-run dataset preparation, training, and validation.

If validation still fails after one retry, the pipeline exits with code 1 and logs `SELF_CORRECTION_EXHAUSTED`. In that case, inspect `logs/pipeline.log` and `logs/validation_metrics.json` to determine whether you need more cleaned training data or a different augmentation strategy.
