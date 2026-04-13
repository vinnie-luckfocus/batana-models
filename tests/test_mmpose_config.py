from pathlib import Path

import pytest
from mmengine import Config

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "rtmpose_m_finetune_baseball.py"


@pytest.fixture(scope="module")
def cfg():
    return Config.fromfile(str(CONFIG_PATH))


def test_config_loadable(cfg):
    assert cfg is not None


def test_nineteen_keypoints(cfg):
    assert cfg.model.head.out_channels == 19


def test_frozen_stages(cfg):
    assert cfg.model.backbone.frozen_stages == 3


def test_optimizer_settings(cfg):
    assert cfg.optim_wrapper.optimizer.type == "AdamW"
    assert cfg.optim_wrapper.optimizer.lr == 5e-4
    assert cfg.optim_wrapper.optimizer.weight_decay == 1e-4


def test_batch_size(cfg):
    assert cfg.train_dataloader.batch_size == 16
