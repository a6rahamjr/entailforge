# Contributing

Create a virtual environment and install the development dependencies:

```bash
pip install -e ".[dev]"
```

Before opening a pull request:

```bash
entailforge check
pytest
```

Keep generated data, model checkpoints, and logs out of commits. Update the
configuration and documentation when behavior changes, and include measured
metrics for model or training changes.
