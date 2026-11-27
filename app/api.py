"""FastAPI scoring endpoint (/health, /predict) integrated with frozen pipelines."""
from typing import Any
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.scoring import UI_MODELS, default_applicant, score_dataframe

app = FastAPI(title="Loan Default Prediction API", version="1.0.0")


class PredictRequest(BaseModel):
    model: str = Field(default="stack_lr (Champion, calibrated OOF)", description="Model key from UI_MODELS")
    cost_ratio: float = Field(default=10.0, ge=1.0, le=50.0)
    records: list[dict[str, Any]] = Field(default_factory=lambda: [default_applicant()])


@app.get("/health")
def health():
    return {"status": "ok", "models": list(UI_MODELS.keys())}


@app.post("/predict")
def predict(req: PredictRequest):
    if req.model not in UI_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model {req.model!r}. Choose from {list(UI_MODELS)}")
    try:
        df = pd.DataFrame(req.records)
        probs = score_dataframe(df, req.model, apply_platt=True)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    thr = 1.0 / (1.0 + req.cost_ratio)
    return {
        "model": req.model,
        "cost_ratio": req.cost_ratio,
        "threshold": thr,
        "predictions": [
            {"p_default": float(p), "decision": "APPROVE" if p < thr else "DECLINE"}
            for p in probs
        ],
    }
