# EntailForge

EntailForge is an offline NLP system for training, evaluating, exporting, and
serving compact logical entailment models.

**Tagline:** Verifiable reasoning without hosted models or external data APIs.

## Overview

EntailForge generates balanced syllogistic datasets and trains a Transformer
encoder from random initialization. It is designed as a small production ML
system rather than a notebook experiment: configuration, preprocessing,
training, evaluation, inference, export, and serving are separate modules with
automated tests.

## Features

- deterministic synthetic train, validation, and test splits;
- strict JSONL schemas and split isolation;
- train-only vocabulary construction;
- token, position, and premise/hypothesis segment embeddings;
- structural chain and converse features fused with the text encoder;
- Transformer encoder trained from scratch;
- label smoothing, gradient clipping, cosine decay, and early stopping;
- accuracy, precision, recall, macro F1, confusion matrix, and calibration;
- premise-level occlusion explanations;
- PyTorch exported-program serialization;
- FastAPI inference service;
- Docker and GitHub Actions support.

## Architecture

```text
entailforge/
|-- configs/
|   `-- default.yaml
|-- src/entailforge/
|   |-- api.py
|   |-- data/
|   |   |-- dataset.py
|   |   |-- generator.py
|   |   |-- preprocessing.py
|   |   `-- schema.py
|   |-- evaluation/
|   |   |-- evaluator.py
|   |   `-- metrics.py
|   |-- inference/
|   |   |-- export.py
|   |   `-- predictor.py
|   |-- models/
|   |   |-- encoder.py
|   |   `-- factory.py
|   |-- training/
|   |   |-- engine.py
|   |   `-- pipeline.py
|   |-- utils/
|   `-- cli.py
|-- tests/
`-- PRD.md
```

The complete product requirements are in [PRD.md](PRD.md).

## Requirements

- Python 3.10 or newer
- PyTorch 2.2 or newer
- A GPU is optional

No API key is required.

## Installation

```bash
git clone <repository-url>
cd entailforge
python -m venv .venv
```

Activate the environment:

```powershell
# Windows
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS or Linux
source .venv/bin/activate
```

Install the project:

```bash
pip install -e ".[dev]"
```

Validate the configuration:

```bash
entailforge check
```

## Training

Generate the data explicitly:

```bash
entailforge generate
```

Train and evaluate:

```bash
entailforge train
```

The training command generates missing data automatically. Outputs are written
to:

- `data/processed/` for JSONL splits;
- `artifacts/best_model.pt` for the best checkpoint;
- `artifacts/training_history.json` for epoch metrics;
- `artifacts/metrics.json` for validation, test, calibration, and baseline
  metrics;
- `artifacts/logs/entailforge.log` for structured run logs.

Edit [configs/default.yaml](configs/default.yaml) to change the dataset, model,
or training parameters.

## Evaluation

```bash
entailforge evaluate --checkpoint artifacts/best_model.pt
```

The report includes accuracy, macro precision, macro recall, macro F1,
confusion matrix, loss, expected calibration error, and temperature.

### Default Benchmark

Measured on June 8, 2026 using the committed default configuration on CPU:

| Metric | Result |
| --- | ---: |
| Test accuracy | 100% |
| Test macro F1 | 1.000 |
| Expected calibration error | 0.0011 |
| Majority baseline accuracy | 50.42% |
| Accuracy gain over baseline | 49.58 points |
| Best epoch | 1 |

The run used 1,200 training, 240 validation, and 240 test examples. Generated
artifacts are intentionally excluded from Git.

## Inference

```bash
entailforge predict \
  --premise "All poets are readers." \
  --premise "No readers are silent." \
  --hypothesis "No poets are silent." \
  --explain
```

The response contains a label, confidence, class probabilities, and optional
premise-level confidence-drop scores.

## Model Export

```bash
entailforge export \
  --checkpoint artifacts/best_model.pt \
  --output artifacts/entailforge.pt2
```

This creates a PyTorch exported program and a JSON metadata file containing the
vocabulary, sequence length, and calibration temperature.

## API

Start the service:

```bash
entailforge serve
```

Or run Uvicorn directly:

```bash
uvicorn entailforge.api:app --host 0.0.0.0 --port 8000
```

Request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "premises": [
      "All poets are readers.",
      "No readers are silent."
    ],
    "hypothesis": "No poets are silent.",
    "explain": true
  }'
```

Health check:

```bash
curl http://localhost:8000/health
```

## Docker

Build and run:

```bash
docker build -t entailforge .
docker run --rm -p 8000:8000 \
  -v "$(pwd)/artifacts:/app/artifacts" \
  entailforge
```

## Testing

```bash
pytest
```

Tests cover deterministic dataset generation, split isolation, preprocessing,
model forward/backward behavior, the training pipeline, checkpoint inference,
model export, and API inference.

## Reproducibility

The configured seed controls Python, NumPy, PyTorch, data generation, and
training-loader shuffling. The default data splits are generated independently
and checked for duplicate records.

## License

Released under the [MIT License](LICENSE).
