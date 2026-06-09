"""PyTorch datasets for encoded entailment examples."""

from typing import Sequence

import torch
from torch.utils.data import Dataset

from entailforge.data.preprocessing import Vocabulary, encode_example
from entailforge.data.schema import LogicExample


class EntailmentDataset(Dataset):
    def __init__(
        self,
        examples: Sequence[LogicExample],
        vocabulary: Vocabulary,
        max_length: int,
    ):
        self.encoded = [
            encode_example(example, vocabulary, max_length) for example in examples
        ]

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, index: int):
        item = self.encoded[index]
        return {
            "input_ids": torch.tensor(item.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(item.attention_mask, dtype=torch.bool),
            "segment_ids": torch.tensor(item.segment_ids, dtype=torch.long),
            "logic_features": torch.tensor(
                item.logic_features,
                dtype=torch.float32,
            ),
            "labels": torch.tensor(item.label, dtype=torch.long),
        }
