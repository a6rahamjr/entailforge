"""Model evaluation and temperature calibration."""

from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from entailforge.evaluation.metrics import (
    classification_metrics,
    expected_calibration_error,
)


@torch.no_grad()
def collect_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for batch in loader:
        labels = batch["labels"].to(device)
        logits = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["segment_ids"].to(device),
            batch["logic_features"].to(device),
        )
        total_loss += criterion(logits, labels).item() * labels.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    return logits, labels, total_loss / len(labels)


def find_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Select a scalar temperature by validation negative log likelihood."""
    criterion = nn.CrossEntropyLoss()
    candidates = torch.linspace(0.5, 3.0, steps=51)
    losses = [
        criterion(logits / candidate, labels).item() for candidate in candidates
    ]
    best_index = int(np.argmin(losses))
    return float(candidates[best_index].item())


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    temperature: float = 1.0,
) -> Dict[str, float | list]:
    logits, labels, loss = collect_logits(model, loader, device)
    probabilities = torch.softmax(logits / temperature, dim=1).numpy()
    predictions = probabilities.argmax(axis=1)
    metrics = classification_metrics(labels.numpy(), predictions)
    metrics["loss"] = float(loss)
    metrics["expected_calibration_error"] = expected_calibration_error(
        labels.numpy(),
        probabilities,
    )
    metrics["temperature"] = float(temperature)
    return metrics
