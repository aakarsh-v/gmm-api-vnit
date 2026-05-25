"""Pydantic request/response models for the alloy prediction API."""
from typing import Optional

from pydantic import BaseModel, Field

from synthetic_generation_core import INPUT_COLS


class CompositionRequest(BaseModel):
    Al: float = 0.0
    Si: float = 0.0
    Fe: float = 0.0
    Cu: float = 0.0
    Mn: float = 0.0
    Mg: float = 0.0
    Cr: float = 0.0
    Ni: float = 0.0
    Zn: float = 0.0
    Ga: float = 0.0
    V: float = 0.0
    Ti: float = 0.0

    def to_dict(self) -> dict:
        return {col: getattr(self, col) for col in INPUT_COLS}


class ForwardPredictResponse(BaseModel):
    composition: dict[str, float]
    predictions: dict[str, float]


class BackwardSearchRequest(BaseModel):
    targets: dict[str, float]
    top_k: int = Field(default=3, ge=1, le=50)


class BackwardCandidate(BaseModel):
    composition: dict[str, float]
    properties: dict[str, float]
    recipe: str
    total_error: float


class BackwardSearchResponse(BaseModel):
    targets: dict[str, float]
    candidates: list[BackwardCandidate]


class PairSearchRequest(BaseModel):
    property_a: str
    value_a: float
    property_b: str
    value_b: float
    tolerance: float = Field(default=0.05, ge=0.0, le=1.0)


class PairSearchResponse(BaseModel):
    found: bool
    candidate: Optional[dict] = None
    message: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    forward_models_loaded: bool
    backward_pool_loaded: bool
    forward_targets: Optional[int] = None
    pool_rows: Optional[int] = None
