# EntailForge Product Requirements

## Problem Statement

Small reasoning projects often depend on hosted language models, loosely
structured prompts, and expensive reinforcement-learning loops. That makes
experiments difficult to reproduce and hard to deploy in constrained
environments.

EntailForge provides a compact, fully offline system for training and serving a
logical entailment model. It generates deterministic reasoning datasets, trains
a Transformer encoder from random initialization, evaluates calibrated
predictions, and exports a deployable model artifact.

## Objective

Build a reproducible logical entailment pipeline that can:

- generate balanced training, validation, and test data;
- train a compact neural model without external model or data APIs;
- report classification and calibration metrics;
- run batch or single-example inference;
- export the trained model for production use;
- expose inference through a small HTTP API.

## Target Users

- ML engineers testing reasoning architectures;
- students learning end-to-end NLP systems;
- researchers who need a deterministic baseline;
- application teams that need a small offline entailment service.

## ML Task Type

Binary natural-language inference over constrained syllogistic statements.

The model predicts whether a hypothesis follows from two premises:

- `1`: entailed
- `0`: not entailed

## Input And Output

### Training Input

JSON Lines records:

```json
{
  "id": "train-000001",
  "premises": [
    "All poets are readers.",
    "No reader is silent."
  ],
  "hypothesis": "No poet is silent.",
  "label": 1,
  "rule": "all_no_chain"
}
```

### Inference Input

- two premise strings;
- one hypothesis string.

### Inference Output

```json
{
  "label": "entailed",
  "confidence": 0.94,
  "probabilities": {
    "not_entailed": 0.06,
    "entailed": 0.94
  }
}
```

## System Workflow

1. Load and validate the YAML configuration.
2. Seed Python, NumPy, and PyTorch.
3. Generate balanced examples from deterministic logic templates.
4. Split data by seed into train, validation, and test files.
5. Build a vocabulary from the training split only.
6. Encode premises and hypothesis with token, position, and segment IDs.
7. Train a Transformer encoder with AdamW and cosine decay.
8. Select the best checkpoint using validation F1.
9. Evaluate accuracy, precision, recall, F1, confusion matrix, and expected
   calibration error.
10. Export the model as a PyTorch exported program.
11. Load the same checkpoint in the CLI or FastAPI service.

## MVP Features

- deterministic synthetic dataset generation;
- schema validation and JSONL storage;
- train-only vocabulary construction;
- compact Transformer encoder;
- seeded training;
- checkpointing and early stopping;
- evaluation report in JSON;
- command-line generation, training, evaluation, prediction, and export;
- unit and integration tests.

## Advanced Features

- curriculum levels for two-step and distractor reasoning;
- temperature scaling for confidence calibration;
- hyperparameter search;
- ONNX export;
- experiment tracking backend;
- additional logical forms and multilingual templates;
- model registry and versioned deployment.

## Success Metrics

- test accuracy of at least 90% on the default synthetic task;
- macro F1 of at least 0.90;
- expected calibration error below 0.15;
- deterministic dataset generation for a fixed seed;
- CPU inference below 50 ms per example on a typical laptop;
- all automated tests pass in CI.

## Constraints

- Python 3.10 or newer;
- no hosted model or dataset API;
- CPU training must remain practical for the default configuration;
- generated splits must not share identical records;
- artifacts must be loadable without training code changes;
- user input must be validated before inference.

## Technology Stack

- Python
- PyTorch
- NumPy
- PyYAML
- FastAPI and Uvicorn
- Pytest
- PyTorch Export
- Docker and GitHub Actions
