import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# Allow importing scripts package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_2d_pose import (
    LOCKED_CLI_FLAGS,
    _copy_config_snapshot,
    _reject_locked_cli_args,
    _run_dummy_training,
    _simulate_epoch_loss,
    _validate_locked_hyperparameters,
)


class TestDummyTraining:
    def test_dummy_training_completes(self, tmp_path: Path) -> None:
        config_path = tmp_path / "rtmpose_m_finetune_baseball.py"
        config_path.write_text("max_epochs = 50\n", encoding="utf-8")
        work_dir = tmp_path / "logs" / "training_logs"
        checkpoint_path = tmp_path / "models" / "2d_pose" / "rtmpose_m_finetuned.pth"

        log = _run_dummy_training(
            config_path=config_path,
            work_dir=work_dir,
            max_epochs=50,
            patience=50,
            reduce_lr_patience=5,
            checkpoint_path=checkpoint_path,
        )

        log_path = work_dir / "training_log.json"
        snapshot = work_dir / "rtmpose_m_finetune_baseball_snapshot.py"

        assert checkpoint_path.is_file()
        assert log_path.is_file()
        assert snapshot.is_file()

        assert "epochs" in log
        assert "train_loss" in log
        assert "val_loss" in log
        assert "lr_history" in log
        assert "best_epoch" in log
        assert "stopped_early" in log

        # With patience larger than max_epochs, early stopping should not trigger.
        assert log["stopped_early"] is False
        assert log["epochs"] == 50

    def test_early_stopping_triggers_on_plateau(self, tmp_path: Path) -> None:
        config_path = tmp_path / "rtmpose_m_finetune_baseball.py"
        config_path.write_text("max_epochs = 50\n", encoding="utf-8")
        work_dir = tmp_path / "logs" / "training_logs"
        checkpoint_path = tmp_path / "models" / "2d_pose" / "rtmpose_m_finetuned.pth"

        # The dummy loss plateaus after epoch 3. With patience=10, early stopping
        # should trigger at epoch 13 (3 improving + 10 plateau).
        log = _run_dummy_training(
            config_path=config_path,
            work_dir=work_dir,
            max_epochs=50,
            patience=10,
            reduce_lr_patience=50,
            checkpoint_path=checkpoint_path,
        )

        assert log["stopped_early"] is True
        assert log["epochs"] == 13

    def test_early_stopping_forced_plateau(self, tmp_path: Path) -> None:
        config_path = tmp_path / "rtmpose_m_finetune_baseball.py"
        config_path.write_text("max_epochs = 100\n", encoding="utf-8")
        work_dir = tmp_path / "logs" / "training_logs"
        checkpoint_path = tmp_path / "models" / "2d_pose" / "rtmpose_m_finetuned.pth"

        log = _run_dummy_training(
            config_path=config_path,
            work_dir=work_dir,
            max_epochs=20,
            patience=3,
            reduce_lr_patience=50,
            checkpoint_path=checkpoint_path,
        )

        # Dummy loss plateaus after epoch 3; with patience=3 it stops at epoch 6.
        assert log["stopped_early"] is True
        assert log["epochs"] == 6


class TestReduceLROnPlateau:
    def test_reduce_lr_on_plateau(self, tmp_path: Path) -> None:
        config_path = tmp_path / "rtmpose_m_finetune_baseball.py"
        config_path.write_text("max_epochs = 50\n", encoding="utf-8")
        work_dir = tmp_path / "logs" / "training_logs"

        log = _run_dummy_training(
            config_path=config_path,
            work_dir=work_dir,
            max_epochs=50,
            patience=50,
            reduce_lr_patience=5,
            lr_factor=0.1,
            lr_min=1e-5,
            lr_max=1e-3,
        )

        lr_history = log["lr_history"]
        initial_lr = 5e-4
        assert lr_history[0] == initial_lr

        # After 5 epochs of no improvement, LR should drop by factor 0.1.
        reduced_lr = initial_lr * 0.1
        assert reduced_lr in lr_history or any(
            abs(lr - reduced_lr) < 1e-12 for lr in lr_history
        )

        # LR should never drop below lr_min.
        assert min(lr_history) >= 1e-5
        # LR should never exceed lr_max.
        assert max(lr_history) <= 1e-3


class TestLockedHyperparameters:
    def test_locked_hyperparameters_rejected_via_cli_subprocess(self, tmp_path: Path) -> None:
        config_path = tmp_path / "rtmpose_m_finetune_baseball.py"
        config_path.write_text("max_epochs = 50\n", encoding="utf-8")
        work_dir = tmp_path / "logs" / "training_logs"

        script = Path(__file__).parent.parent / "scripts" / "train_2d_pose.py"
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--config",
                str(config_path),
                "--work-dir",
                str(work_dir),
                "--batch-size",
                "32",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "not allowed" in result.stderr.lower() or "not allowed" in result.stdout.lower()

    def test_locked_hyperparameters_config_mismatch(self, tmp_path: Path) -> None:
        class FakeCfg:
            model = {"backbone": {"frozen_stages": 0}}
            optim_wrapper = {"optimizer": {"lr": 1e-2, "weight_decay": 1e-3}}
            train_dataloader = {"batch_size": 32}
            train_cfg = {"max_epochs": 100}

        with pytest.raises(ValueError) as exc_info:
            _validate_locked_hyperparameters(FakeCfg())
        msg = str(exc_info.value).lower()
        assert "frozen_stages" in msg
        assert "lr" in msg
        assert "batch_size" in msg

    def test_reject_locked_cli_flags_directly(self) -> None:
        for flag in LOCKED_CLI_FLAGS:
            with pytest.raises(ValueError) as exc_info:
                _reject_locked_cli_args(None, [flag, "32"])
            assert "not allowed" in str(exc_info.value).lower()


class TestHelpers:
    def test_simulate_epoch_loss_bounds(self) -> None:
        for epoch in range(1, 51):
            t, v = _simulate_epoch_loss(epoch)
            assert t >= 0.0
            assert v >= 0.0

    def test_copy_config_snapshot(self, tmp_path: Path) -> None:
        src = tmp_path / "cfg.py"
        src.write_text("a = 1\n", encoding="utf-8")
        dest_dir = tmp_path / "out"
        snapshot = _copy_config_snapshot(src, dest_dir)
        assert snapshot.is_file()
        assert snapshot.read_text(encoding="utf-8") == "a = 1\n"
