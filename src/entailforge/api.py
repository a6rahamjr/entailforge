"""FastAPI inference service."""

import os
from functools import lru_cache
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from entailforge.inference.predictor import Predictor


app = FastAPI(
    title="EntailForge API",
    version="0.1.0",
    description="Logical entailment inference",
)


class PredictionRequest(BaseModel):
    premises: List[str] = Field(min_length=2, max_length=5)
    hypothesis: str = Field(min_length=3, max_length=500)
    explain: bool = False


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict
    premise_importance: list | None = None


def _checkpoint_path() -> Path:
    return Path(
        os.getenv(
            "ENTAILFORGE_CHECKPOINT",
            "artifacts/best_model.pt",
        )
    )


@lru_cache(maxsize=1)
def _get_predictor() -> Predictor:
    checkpoint = _checkpoint_path()
    if not checkpoint.is_file():
        raise FileNotFoundError(str(checkpoint))
    return Predictor(checkpoint)


@app.get("/health")
def health():
    checkpoint = _checkpoint_path()
    return {
        "status": "ready" if checkpoint.is_file() else "model_missing",
        "checkpoint": str(checkpoint),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        predictor = _get_predictor()
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=f"Model checkpoint not found: {error}",
        ) from error
    result = predictor.predict(
        premises=request.premises,
        hypothesis=request.hypothesis,
        explain=request.explain,
    )
    result.setdefault("premise_importance", None)
    return result
