import json
from pathlib import Path


def test_training_pipeline_creates_artifacts(tiny_training_run):
    config, report = tiny_training_run

    assert Path(report["checkpoint"]).is_file()
    assert (config.artifact_dir / "metrics.json").is_file()
    assert (config.artifact_dir / "training_history.json").is_file()
    assert report["dataset"] == {
        "train": 64,
        "validation": 16,
        "test": 16,
    }

    saved = json.loads(
        (config.artifact_dir / "metrics.json").read_text(encoding="utf-8")
    )
    assert "macro_f1" in saved["test"]
    assert "accuracy_gain_over_baseline" in saved
