from pathlib import Path

import pytest

from entailforge.training.pipeline import run_training
from entailforge.utils.config import (
    AppConfig,
    DataConfig,
    ModelConfig,
    TrainingConfig,
)


@pytest.fixture(scope="session")
def tiny_training_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("entailforge")
    config = AppConfig(
        project_name="EntailForgeTest",
        artifact_dir=root / "artifacts",
        data_dir=root / "data",
        seed=17,
        data=DataConfig(
            train_size=64,
            validation_size=16,
            test_size=16,
            max_length=48,
            min_token_frequency=1,
            include_distractors=False,
        ),
        model=ModelConfig(
            embedding_dim=16,
            hidden_dim=32,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
            num_classes=2,
        ),
        training=TrainingConfig(
            epochs=2,
            batch_size=16,
            learning_rate=0.003,
            weight_decay=0.0,
            label_smoothing=0.0,
            gradient_clip_norm=1.0,
            early_stopping_patience=2,
            num_workers=0,
        ),
        api={"host": "127.0.0.1", "port": 8000},
    )
    report = run_training(config, force_data=True, device_name="cpu")
    return config, report
