from pathlib import Path

import pytest
import torch

from entailforge.inference.export import export_model
from entailforge.inference.predictor import Predictor


def test_checkpoint_inference_and_export(tiny_training_run):
    _, report = tiny_training_run
    checkpoint = Path(report["checkpoint"])
    predictor = Predictor(checkpoint, device="cpu")

    result = predictor.predict(
        premises=[
            "All poets are readers.",
            "No readers are silent.",
        ],
        hypothesis="No poets are silent.",
        explain=True,
    )

    assert result["label"] in {"entailed", "not_entailed"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)
    assert len(result["premise_importance"]) == 2

    output = checkpoint.parent / "model.pt2"
    export_model(checkpoint, output)
    exported = torch.export.load(str(output))
    assert output.is_file()
    assert exported is not None
