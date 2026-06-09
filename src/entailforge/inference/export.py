"""PyTorch exported-program serialization."""

import json
from pathlib import Path

import torch

from entailforge.inference.predictor import Predictor


def export_model(
    checkpoint_path: Path,
    output_path: Path,
) -> Path:
    predictor = Predictor(checkpoint_path, device="cpu")
    model = predictor.model.cpu().eval()
    max_length = predictor.max_length
    example_inputs = (
        torch.zeros((1, max_length), dtype=torch.long),
        torch.ones((1, max_length), dtype=torch.bool),
        torch.zeros((1, max_length), dtype=torch.long),
        torch.zeros(
            (1, predictor.model.spec.logic_feature_dim),
            dtype=torch.float32,
        ),
    )
    exported = torch.export.export(model, example_inputs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(exported, str(output_path))

    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "format": "torch.export",
                "max_length": max_length,
                "temperature": predictor.temperature,
                "vocabulary": predictor.vocabulary.token_to_id,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path
