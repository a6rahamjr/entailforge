"""Production-style training loop for EntailForge."""

import json
import logging
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from entailforge.data.preprocessing import Vocabulary
from entailforge.evaluation.evaluator import (
    collect_logits,
    evaluate_model,
    find_temperature,
)
from entailforge.models.encoder import EntailmentTransformer
from entailforge.utils.config import TrainingConfig


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingResult:
    checkpoint_path: Path
    history_path: Path
    best_epoch: int
    validation_metrics: Dict[str, float | list]
    temperature: float


class Trainer:
    def __init__(
        self,
        model: EntailmentTransformer,
        training_config: TrainingConfig,
        vocabulary: Vocabulary,
        max_length: int,
        artifact_dir: Path,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.config = training_config
        self.vocabulary = vocabulary
        self.max_length = max_length
        self.artifact_dir = artifact_dir
        self.device = device
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
    ) -> TrainingResult:
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, self.config.epochs * len(train_loader)),
        )
        criterion = nn.CrossEntropyLoss(
            label_smoothing=self.config.label_smoothing,
        )

        history: List[Dict[str, float]] = []
        best_state = None
        best_f1 = -1.0
        best_epoch = 0
        epochs_without_improvement = 0
        use_amp = self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            running_loss = 0.0
            sample_count = 0

            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                labels = batch["labels"].to(self.device)
                autocast = (
                    torch.amp.autocast(device_type="cuda")
                    if use_amp
                    else nullcontext()
                )
                with autocast:
                    logits = self.model(
                        batch["input_ids"].to(self.device),
                        batch["attention_mask"].to(self.device),
                        batch["segment_ids"].to(self.device),
                        batch["logic_features"].to(self.device),
                    )
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm,
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                running_loss += loss.item() * labels.size(0)
                sample_count += labels.size(0)

            validation = evaluate_model(
                self.model,
                validation_loader,
                self.device,
            )
            epoch_record = {
                "epoch": float(epoch),
                "train_loss": running_loss / sample_count,
                "validation_loss": float(validation["loss"]),
                "validation_accuracy": float(validation["accuracy"]),
                "validation_macro_f1": float(validation["macro_f1"]),
                "learning_rate": float(scheduler.get_last_lr()[0]),
            }
            history.append(epoch_record)
            LOGGER.info(
                "epoch=%s train_loss=%.4f val_loss=%.4f val_f1=%.4f",
                epoch,
                epoch_record["train_loss"],
                epoch_record["validation_loss"],
                epoch_record["validation_macro_f1"],
            )

            if validation["macro_f1"] > best_f1 + 1e-6:
                best_f1 = float(validation["macro_f1"])
                best_epoch = epoch
                best_state = deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if (
                    epochs_without_improvement
                    >= self.config.early_stopping_patience
                ):
                    LOGGER.info("early stopping at epoch %s", epoch)
                    break

        if best_state is None:
            raise RuntimeError("Training did not produce a checkpoint.")

        self.model.load_state_dict(best_state)
        validation_logits, validation_labels, _ = collect_logits(
            self.model,
            validation_loader,
            self.device,
        )
        temperature = find_temperature(validation_logits, validation_labels)
        validation_metrics = evaluate_model(
            self.model,
            validation_loader,
            self.device,
            temperature=temperature,
        )

        checkpoint_path = self.artifact_dir / "best_model.pt"
        history_path = self.artifact_dir / "training_history.json"
        torch.save(
            {
                "format_version": 1,
                "model_state": self.model.state_dict(),
                "model_spec": self.model.spec.to_dict(),
                "vocabulary": self.vocabulary.token_to_id,
                "max_length": self.max_length,
                "temperature": temperature,
                "best_epoch": best_epoch,
                "validation_metrics": validation_metrics,
                "training_config": asdict(self.config),
            },
            checkpoint_path,
        )
        history_path.write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )

        return TrainingResult(
            checkpoint_path=checkpoint_path,
            history_path=history_path,
            best_epoch=best_epoch,
            validation_metrics=validation_metrics,
            temperature=temperature,
        )
