"""Reproducible synthetic data for logical entailment."""

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from entailforge.data.schema import LogicExample


NOUNS = (
    "artists",
    "bakers",
    "builders",
    "chefs",
    "clerks",
    "dancers",
    "designers",
    "doctors",
    "drivers",
    "editors",
    "farmers",
    "gardeners",
    "judges",
    "lawyers",
    "musicians",
    "nurses",
    "painters",
    "pilots",
    "poets",
    "readers",
    "runners",
    "sailors",
    "scientists",
    "singers",
    "teachers",
    "writers",
)


def _all(left: str, right: str) -> str:
    return f"All {left} are {right}."


def _no(left: str, right: str) -> str:
    return f"No {left} are {right}."


def _sample_distinct(rng: random.Random, count: int) -> Tuple[str, ...]:
    return tuple(rng.sample(NOUNS, count))


def _build_positive(rng: random.Random) -> Tuple[List[str], str, str]:
    rule = rng.choice(("all_chain", "exclusion_chain", "inverse_exclusion"))
    a, b, c = _sample_distinct(rng, 3)

    if rule == "all_chain":
        return [_all(a, b), _all(b, c)], _all(a, c), rule
    if rule == "exclusion_chain":
        return [_all(a, b), _no(b, c)], _no(a, c), rule
    return [_no(a, b), _all(c, a)], _no(c, b), rule


def _build_negative(rng: random.Random) -> Tuple[List[str], str, str]:
    rule = rng.choice(("contradiction", "invalid_converse", "unrelated_target"))
    a, b, c, d = _sample_distinct(rng, 4)

    if rule == "contradiction":
        return [_all(a, b), _no(b, c)], _all(a, c), rule
    if rule == "invalid_converse":
        return [_all(a, b), _all(b, c)], _all(c, a), rule
    return [_all(a, b), _all(b, c)], _all(a, d), rule


def _add_distractor(
    rng: random.Random,
    premises: List[str],
    used_words: Sequence[str],
) -> List[str]:
    choices = [noun for noun in NOUNS if noun not in used_words]
    left, right = rng.sample(choices, 2)
    return [*premises, _all(left, right)]


def generate_examples(
    size: int,
    seed: int,
    split: str,
    include_distractors: bool = True,
) -> List[LogicExample]:
    """Generate a balanced, unique split."""
    if size < 2:
        raise ValueError("Dataset size must be at least 2.")

    rng = random.Random(seed)
    target_positive = size // 2
    labels = [1] * target_positive + [0] * (size - target_positive)
    rng.shuffle(labels)

    examples: List[LogicExample] = []
    seen = set()
    attempts = 0

    while len(examples) < size:
        if attempts > size * 100:
            raise RuntimeError("Unable to generate enough unique examples.")
        attempts += 1

        label = labels[len(examples)]
        premises, hypothesis, rule = (
            _build_positive(rng) if label == 1 else _build_negative(rng)
        )
        if include_distractors and rng.random() < 0.35:
            used_words = " ".join([*premises, hypothesis]).lower().split()
            premises = _add_distractor(rng, premises, used_words)

        signature = (tuple(premises), hypothesis, label)
        if signature in seen:
            continue
        seen.add(signature)

        examples.append(
            LogicExample(
                id=f"{split}-{len(examples):06d}",
                premises=premises,
                hypothesis=hypothesis,
                label=label,
                rule=rule,
            )
        )

    return examples


def write_jsonl(path: Path, examples: Iterable[LogicExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for example in examples:
            output.write(json.dumps(example.to_dict(), sort_keys=True) + "\n")


def load_jsonl(path: Path) -> List[LogicExample]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    examples = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                examples.append(LogicExample.from_dict(payload))
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid record in {path} at line {line_number}: {error}"
                ) from error
    if not examples:
        raise ValueError(f"Dataset is empty: {path}")
    return examples


def generate_dataset_splits(
    output_dir: Path,
    train_size: int,
    validation_size: int,
    test_size: int,
    seed: int,
    include_distractors: bool = True,
) -> Dict[str, Path]:
    """Generate independent deterministic splits."""
    sizes = {
        "train": train_size,
        "validation": validation_size,
        "test": test_size,
    }
    paths = {}
    all_signatures = set()

    for offset, (split, size) in enumerate(sizes.items()):
        examples = generate_examples(
            size=size,
            seed=seed + (offset * 1009),
            split=split,
            include_distractors=include_distractors,
        )
        filtered = []
        for example in examples:
            signature = (tuple(example.premises), example.hypothesis, example.label)
            if signature not in all_signatures:
                all_signatures.add(signature)
                filtered.append(example)

        if len(filtered) != size:
            replacements = generate_examples(
                size=(size - len(filtered)) * 4 + 2,
                seed=seed + (offset * 1009) + 503,
                split=split,
                include_distractors=include_distractors,
            )
            for candidate in replacements:
                signature = (
                    tuple(candidate.premises),
                    candidate.hypothesis,
                    candidate.label,
                )
                if signature in all_signatures:
                    continue
                all_signatures.add(signature)
                filtered.append(
                    LogicExample(
                        id=f"{split}-{len(filtered):06d}",
                        premises=candidate.premises,
                        hypothesis=candidate.hypothesis,
                        label=candidate.label,
                        rule=candidate.rule,
                    )
                )
                if len(filtered) == size:
                    break

        if len(filtered) != size:
            raise RuntimeError(f"Could not create a unique {split} split.")

        path = output_dir / f"{split}.jsonl"
        write_jsonl(path, filtered)
        paths[split] = path

    return paths
