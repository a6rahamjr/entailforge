"""Typed configuration loading."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class DataConfig:
    train_size: int
    validation_size: int
    test_size: int
    max_length: int
    min_token_frequency: int
    include_distractors: bool


@dataclass(frozen=True)
class ModelConfig:
    embedding_dim: int
    hidden_dim: int
    num_heads: int
    num_layers: int
    dropout: float
    num_classes: int


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    label_smoothing: float
    gradient_clip_norm: float
    early_stopping_patience: int
    num_workers: int


@dataclass(frozen=True)
class AppConfig:
    project_name: str
    artifact_dir: Path
    data_dir: Path
    seed: int
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    api: Dict[str, Any]


def _positive(value: Any, name: str) -> None:
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be positive.")


def load_config(path: Path) -> AppConfig:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as source:
        payload = yaml.safe_load(source)
    if not isinstance(payload, dict):
        raise ValueError("Configuration root must be a mapping.")

    for section in ("project", "data", "model", "training", "api"):
        if not isinstance(payload.get(section), dict):
            raise ValueError(f"Missing configuration section: {section}")

    project = payload["project"]
    data = DataConfig(**payload["data"])
    model = ModelConfig(**payload["model"])
    training = TrainingConfig(**payload["training"])

    for name, value in (
        ("data.train_size", data.train_size),
        ("data.validation_size", data.validation_size),
        ("data.test_size", data.test_size),
        ("data.max_length", data.max_length),
        ("model.embedding_dim", model.embedding_dim),
        ("model.hidden_dim", model.hidden_dim),
        ("model.num_heads", model.num_heads),
        ("model.num_layers", model.num_layers),
        ("training.epochs", training.epochs),
        ("training.batch_size", training.batch_size),
        ("training.learning_rate", training.learning_rate),
    ):
        _positive(value, name)

    if model.embedding_dim % model.num_heads:
        raise ValueError("model.embedding_dim must be divisible by model.num_heads.")
    if model.num_classes != 2:
        raise ValueError("EntailForge currently requires model.num_classes=2.")
    if not 0 <= model.dropout < 1:
        raise ValueError("model.dropout must be in [0, 1).")
    if not 0 <= training.label_smoothing < 1:
        raise ValueError("training.label_smoothing must be in [0, 1).")

    return AppConfig(
        project_name=str(project["name"]),
        artifact_dir=Path(project["artifact_dir"]),
        data_dir=Path(project["data_dir"]),
        seed=int(payload.get("seed", 42)),
        data=data,
        model=model,
        training=training,
        api=dict(payload["api"]),
    )
