"""End-to-end data, training, and evaluation orchestration."""

import json
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from entailforge.data.dataset import EntailmentDataset
from entailforge.data.generator import generate_dataset_splits, load_jsonl
from entailforge.data.preprocessing import Vocabulary
from entailforge.evaluation.evaluator import evaluate_model
from entailforge.inference.predictor import Predictor
from entailforge.models.factory import build_model
from entailforge.training.engine import Trainer
from entailforge.utils.config import AppConfig
from entailforge.utils.device import select_device
from entailforge.utils.logging import configure_logging
from entailforge.utils.seed import seed_everything


def prepare_data(config: AppConfig, force: bool = False) -> Dict[str, Path]:
    paths = {
        split: config.data_dir / f"{split}.jsonl"
        for split in ("train", "validation", "test")
    }
    if force or not all(path.is_file() for path in paths.values()):
        paths = generate_dataset_splits(
            output_dir=config.data_dir,
            train_size=config.data.train_size,
            validation_size=config.data.validation_size,
            test_size=config.data.test_size,
            seed=config.seed,
            include_distractors=config.data.include_distractors,
        )
    return paths


def _build_loaders(config: AppConfig, paths: Dict[str, Path]):
    examples = {split: load_jsonl(path) for split, path in paths.items()}
    vocabulary = Vocabulary.build(
        examples["train"],
        min_frequency=config.data.min_token_frequency,
    )
    datasets = {
        split: EntailmentDataset(
            records,
            vocabulary,
            config.data.max_length,
        )
        for split, records in examples.items()
    }
    generator = torch.Generator().manual_seed(config.seed)
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=config.training.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=config.training.num_workers,
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
        ),
    }
    return examples, vocabulary, loaders


def run_training(
    config: AppConfig,
    force_data: bool = False,
    device_name: str = "auto",
) -> Dict[str, object]:
    logger = configure_logging(config.artifact_dir / "logs")
    seed_everything(config.seed)
    paths = prepare_data(config, force=force_data)
    examples, vocabulary, loaders = _build_loaders(config, paths)
    device = select_device(device_name)
    logger.info("training on device=%s", device)

    model = build_model(config, vocab_size=len(vocabulary))
    trainer = Trainer(
        model=model,
        training_config=config.training,
        vocabulary=vocabulary,
        max_length=config.data.max_length,
        artifact_dir=config.artifact_dir,
        device=device,
    )
    training_result = trainer.train(
        loaders["train"],
        loaders["validation"],
    )
    test_metrics = evaluate_model(
        trainer.model,
        loaders["test"],
        device,
        temperature=training_result.temperature,
    )

    test_labels = [example.label for example in examples["test"]]
    majority_count = max(test_labels.count(0), test_labels.count(1))
    majority_accuracy = majority_count / len(test_labels)
    report: Dict[str, object] = {
        "project": config.project_name,
        "checkpoint": str(training_result.checkpoint_path),
        "best_epoch": training_result.best_epoch,
        "dataset": {
            split: len(records) for split, records in examples.items()
        },
        "vocabulary_size": len(vocabulary),
        "validation": training_result.validation_metrics,
        "test": test_metrics,
        "baseline": {
            "strategy": "majority_class",
            "accuracy": majority_accuracy,
        },
        "accuracy_gain_over_baseline": (
            float(test_metrics["accuracy"]) - majority_accuracy
        ),
    }
    metrics_path = config.artifact_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("metrics saved to %s", metrics_path)
    return report


def run_evaluation(
    config: AppConfig,
    checkpoint_path: Path,
    device_name: str = "auto",
) -> Dict[str, object]:
    paths = prepare_data(config, force=False)
    _, _, loaders = _build_loaders(config, paths)
    predictor = Predictor(checkpoint_path, device=device_name)
    metrics = evaluate_model(
        predictor.model,
        loaders["test"],
        predictor.device,
        temperature=predictor.temperature,
    )
    return metrics
