from entailforge.data.generator import (
    generate_dataset_splits,
    generate_examples,
    load_jsonl,
)
from entailforge.data.preprocessing import Vocabulary, encode_example


def test_generation_is_deterministic_and_balanced():
    first = generate_examples(20, seed=7, split="train")
    second = generate_examples(20, seed=7, split="train")

    assert [item.to_dict() for item in first] == [
        item.to_dict() for item in second
    ]
    assert sum(item.label for item in first) == 10


def test_splits_are_disjoint_and_loadable(tmp_path):
    paths = generate_dataset_splits(
        output_dir=tmp_path,
        train_size=30,
        validation_size=10,
        test_size=10,
        seed=11,
    )
    splits = {name: load_jsonl(path) for name, path in paths.items()}
    signatures = {
        name: {
            (tuple(item.premises), item.hypothesis, item.label)
            for item in examples
        }
        for name, examples in splits.items()
    }

    assert len(splits["train"]) == 30
    assert signatures["train"].isdisjoint(signatures["validation"])
    assert signatures["train"].isdisjoint(signatures["test"])
    assert signatures["validation"].isdisjoint(signatures["test"])


def test_encoding_has_fixed_length_and_segments():
    example = generate_examples(2, seed=5, split="test")[0]
    vocabulary = Vocabulary.build([example])
    encoded = encode_example(example, vocabulary, max_length=32)

    assert len(encoded.input_ids) == 32
    assert len(encoded.attention_mask) == 32
    assert len(encoded.segment_ids) == 32
    assert len(encoded.logic_features) == 8
    assert 1 in encoded.segment_ids
