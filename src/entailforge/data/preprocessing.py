"""Vocabulary building and segment-aware text encoding."""

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from entailforge.data.schema import LogicExample


TOKEN_PATTERN = re.compile(r"[a-z]+|[?.!,;:]", re.IGNORECASE)
RELATION_PATTERN = re.compile(
    r"^(all|no)\s+([a-z]+)\s+are\s+([a-z]+)\.$",
    re.IGNORECASE,
)
SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]")
LOGIC_FEATURE_DIM = 8


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())


@dataclass(frozen=True)
class EncodedExample:
    input_ids: List[int]
    attention_mask: List[int]
    segment_ids: List[int]
    logic_features: List[float]
    label: int


class Vocabulary:
    def __init__(self, token_to_id: Dict[str, int]):
        self.token_to_id = dict(token_to_id)
        self.id_to_token = {
            token_id: token for token, token_id in self.token_to_id.items()
        }
        for token in SPECIAL_TOKENS:
            if token not in self.token_to_id:
                raise ValueError(f"Missing special token: {token}")

    @classmethod
    def build(
        cls,
        examples: Iterable[LogicExample],
        min_frequency: int = 1,
    ) -> "Vocabulary":
        counts = Counter()
        for example in examples:
            for text in [*example.premises, example.hypothesis]:
                counts.update(tokenize(text))

        tokens = list(SPECIAL_TOKENS)
        tokens.extend(
            token
            for token, count in sorted(counts.items())
            if count >= min_frequency and token not in SPECIAL_TOKENS
        )
        return cls({token: index for index, token in enumerate(tokens)})

    def __len__(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id["[PAD]"]

    def lookup(self, token: str) -> int:
        return self.token_to_id.get(token, self.token_to_id["[UNK]"])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.token_to_id, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        return cls(json.loads(path.read_text(encoding="utf-8")))


def encode_example(
    example: LogicExample,
    vocabulary: Vocabulary,
    max_length: int,
) -> EncodedExample:
    if max_length < 8:
        raise ValueError("max_length must be at least 8.")

    premise_tokens: List[str] = []
    for index, premise in enumerate(example.premises):
        if index:
            premise_tokens.append("[SEP]")
        premise_tokens.extend(tokenize(premise))

    tokens = ["[CLS]", *premise_tokens, "[SEP]"]
    segments = [0] * len(tokens)

    hypothesis_tokens = [*tokenize(example.hypothesis), "[SEP]"]
    tokens.extend(hypothesis_tokens)
    segments.extend([1] * len(hypothesis_tokens))

    tokens = tokens[:max_length]
    segments = segments[:max_length]
    attention = [1] * len(tokens)

    padding = max_length - len(tokens)
    tokens.extend(["[PAD]"] * padding)
    segments.extend([0] * padding)
    attention.extend([0] * padding)

    return EncodedExample(
        input_ids=[vocabulary.lookup(token) for token in tokens],
        attention_mask=attention,
        segment_ids=segments,
        logic_features=extract_logic_features(
            example.premises,
            example.hypothesis,
        ),
        label=example.label,
    )


def _parse_relation(text: str):
    match = RELATION_PATTERN.match(text.strip())
    if not match:
        return None
    quantifier, subject, object_ = match.groups()
    return quantifier.lower(), subject.lower(), object_.lower()


def extract_logic_features(
    premises: Sequence[str],
    hypothesis: str,
) -> List[float]:
    """Extract permutation-invariant relation features."""
    relations = [
        parsed for premise in premises if (parsed := _parse_relation(premise))
    ]
    target = _parse_relation(hypothesis)
    if target is None:
        return [0.0] * LOGIC_FEATURE_DIM

    target_quantifier, target_subject, target_object = target
    direct_match = float(target in relations)
    reverse_match = float(
        (target_quantifier, target_object, target_subject) in relations
    )

    forward_all_chain = 0.0
    forward_no_chain = 0.0
    reverse_all_chain = 0.0
    for first_quantifier, first_subject, middle in relations:
        for second_quantifier, second_subject, second_object in relations:
            if middle != second_subject:
                continue
            if (
                first_subject == target_subject
                and second_object == target_object
            ):
                if first_quantifier == "all" and second_quantifier == "all":
                    forward_all_chain = 1.0
                if first_quantifier == "all" and second_quantifier == "no":
                    forward_no_chain = 1.0
            if (
                first_subject == target_object
                and second_object == target_subject
                and first_quantifier == "all"
                and second_quantifier == "all"
            ):
                reverse_all_chain = 1.0

    return [
        float(target_quantifier == "all"),
        float(target_quantifier == "no"),
        direct_match,
        reverse_match,
        forward_all_chain,
        forward_no_chain,
        reverse_all_chain,
        float(len(premises) > 2),
    ]
