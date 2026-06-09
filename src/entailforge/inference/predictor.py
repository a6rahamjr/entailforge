"""Checkpoint-backed inference."""

from pathlib import Path
from typing import Dict, List, Optional

import torch

from entailforge.data.preprocessing import Vocabulary, encode_example
from entailforge.data.schema import LogicExample
from entailforge.models.encoder import EncoderSpec, EntailmentTransformer
from entailforge.utils.device import select_device


class Predictor:
    def __init__(
        self,
        checkpoint_path: Path,
        device: str = "auto",
    ):
        self.device = select_device(device)
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        self.vocabulary = Vocabulary(checkpoint["vocabulary"])
        self.max_length = int(checkpoint["max_length"])
        self.temperature = float(checkpoint.get("temperature", 1.0))
        self.model = EntailmentTransformer(
            EncoderSpec(**checkpoint["model_spec"])
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        premises: List[str],
        hypothesis: str,
        explain: bool = False,
    ) -> Dict[str, object]:
        example = LogicExample(
            id="inference",
            premises=premises,
            hypothesis=hypothesis,
            label=0,
            rule="inference",
        )
        encoded = encode_example(
            example,
            self.vocabulary,
            self.max_length,
        )
        logits = self.model(
            torch.tensor([encoded.input_ids], dtype=torch.long, device=self.device),
            torch.tensor(
                [encoded.attention_mask],
                dtype=torch.bool,
                device=self.device,
            ),
            torch.tensor(
                [encoded.segment_ids],
                dtype=torch.long,
                device=self.device,
            ),
            torch.tensor(
                [encoded.logic_features],
                dtype=torch.float32,
                device=self.device,
            ),
        )
        probabilities = torch.softmax(logits / self.temperature, dim=1)[0]
        prediction = int(probabilities.argmax().item())
        result: Dict[str, object] = {
            "label": "entailed" if prediction == 1 else "not_entailed",
            "confidence": float(probabilities[prediction].item()),
            "probabilities": {
                "not_entailed": float(probabilities[0].item()),
                "entailed": float(probabilities[1].item()),
            },
        }
        if explain:
            result["premise_importance"] = self._premise_importance(
                premises,
                hypothesis,
                prediction,
                float(probabilities[prediction].item()),
            )
        return result

    def _premise_importance(
        self,
        premises: List[str],
        hypothesis: str,
        predicted_class: int,
        full_confidence: float,
    ) -> List[Dict[str, object]]:
        importance = []
        if len(premises) <= 2:
            replacement = "No additional information is provided."
        else:
            replacement = None

        for index, premise in enumerate(premises):
            reduced = [item for offset, item in enumerate(premises) if offset != index]
            if replacement is not None:
                reduced.append(replacement)
            prediction = self.predict(reduced, hypothesis, explain=False)
            reduced_confidence = float(
                prediction["probabilities"][
                    "entailed" if predicted_class == 1 else "not_entailed"
                ]
            )
            importance.append(
                {
                    "premise": premise,
                    "confidence_drop": max(
                        0.0,
                        full_confidence - reduced_confidence,
                    ),
                }
            )
        return sorted(
            importance,
            key=lambda item: float(item["confidence_drop"]),
            reverse=True,
        )
