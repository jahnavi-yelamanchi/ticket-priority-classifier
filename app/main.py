"""Public FastAPI contract for the Triage classifier."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from app.model import ModelNotReadyError, TriageModel

app = FastAPI(title="Triage API", version="0.1.0", description="Support-ticket priority classification.")
model = TriageModel()


class PredictionRequest(BaseModel):
    ticket: str = Field(min_length=1, max_length=10_000, description="Customer support ticket text.")


class PredictionResponse(BaseModel):
    priority: str
    confidence: float
    probabilities: dict[str, float]


def _not_ready(error: ModelNotReadyError) -> HTTPException:
    return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error))


@app.get("/health")
def health() -> dict[str, object]:
    return model.health()


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> dict[str, object]:
    try:
        return model.predict(request.ticket)
    except ModelNotReadyError as error:
        raise _not_ready(error) from error


@app.get("/metrics")
def metrics() -> dict[str, object]:
    try:
        return model.metrics()
    except ModelNotReadyError as error:
        raise _not_ready(error) from error
