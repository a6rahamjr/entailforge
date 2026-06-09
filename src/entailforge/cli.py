"""Command line interface for EntailForge."""

import argparse
import json
from pathlib import Path
from typing import List, Optional

from entailforge.data.generator import generate_dataset_splits
from entailforge.inference.export import export_model
from entailforge.inference.predictor import Predictor
from entailforge.training.pipeline import run_evaluation, run_training
from entailforge.utils.config import load_config


DEFAULT_CONFIG = Path("configs/default.yaml")


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to YAML configuration",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="entailforge",
        description="Train and serve compact logical entailment models",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="Validate configuration")
    _add_config_argument(check_parser)

    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate deterministic dataset splits",
    )
    _add_config_argument(generate_parser)
    generate_parser.add_argument("--force", action="store_true")

    train_parser = subparsers.add_parser("train", help="Train and evaluate a model")
    _add_config_argument(train_parser)
    train_parser.add_argument("--force-data", action="store_true")
    train_parser.add_argument("--device", default="auto")

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a saved checkpoint",
    )
    _add_config_argument(evaluate_parser)
    evaluate_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/best_model.pt"),
    )
    evaluate_parser.add_argument("--device", default="auto")

    predict_parser = subparsers.add_parser(
        "predict",
        help="Run one inference request",
    )
    predict_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/best_model.pt"),
    )
    predict_parser.add_argument(
        "--premise",
        action="append",
        required=True,
        help="Premise text; pass at least twice",
    )
    predict_parser.add_argument("--hypothesis", required=True)
    predict_parser.add_argument("--explain", action="store_true")
    predict_parser.add_argument("--device", default="auto")

    export_parser = subparsers.add_parser(
        "export",
        help="Export a checkpoint as a PyTorch program",
    )
    export_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/best_model.pt"),
    )
    export_parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/entailforge.pt2"),
    )

    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI service")
    _add_config_argument(serve_parser)
    serve_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/best_model.pt"),
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "check":
        config = load_config(args.config)
        print(f"Configuration valid for {config.project_name}")
        return 0

    if args.command == "generate":
        config = load_config(args.config)
        paths = generate_dataset_splits(
            output_dir=config.data_dir,
            train_size=config.data.train_size,
            validation_size=config.data.validation_size,
            test_size=config.data.test_size,
            seed=config.seed,
            include_distractors=config.data.include_distractors,
        )
        print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
        return 0

    if args.command == "train":
        config = load_config(args.config)
        report = run_training(
            config,
            force_data=args.force_data,
            device_name=args.device,
        )
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "evaluate":
        config = load_config(args.config)
        metrics = run_evaluation(
            config,
            args.checkpoint,
            device_name=args.device,
        )
        print(json.dumps(metrics, indent=2))
        return 0

    if args.command == "predict":
        if len(args.premise) < 2:
            raise SystemExit("At least two --premise values are required.")
        predictor = Predictor(args.checkpoint, device=args.device)
        result = predictor.predict(
            args.premise,
            args.hypothesis,
            explain=args.explain,
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "export":
        output = export_model(args.checkpoint, args.output)
        print(f"Exported model to {output}")
        return 0

    if args.command == "serve":
        import os

        import uvicorn

        config = load_config(args.config)
        os.environ["ENTAILFORGE_CHECKPOINT"] = str(args.checkpoint)
        uvicorn.run(
            "entailforge.api:app",
            host=str(config.api["host"]),
            port=int(config.api["port"]),
        )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
