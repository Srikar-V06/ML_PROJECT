"""Smoke tests for the Streamlit scoring helpers and FastAPI endpoint."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api import app
from app.scoring import default_applicant, explain_single, score_dataframe, validate_and_prepare


def test_app_scoring_smoke():
    df = pd.DataFrame([default_applicant()] * 3)
    for model_key in [
        "stack_lr (Champion, calibrated OOF)",
        "catboost (Platt-calibrated)",
        "xgb_hist (Platt-calibrated)",
        "lgbm_native_flags (Platt-calibrated)",
        "logreg_baseline (Linear Baseline)",
    ]:
        p = score_dataframe(df, model_key, apply_platt=True)
        assert p.shape == (3,)
        assert np.all((p > 0.0) & (p < 1.0))


def test_app_shap_smoke():
    df = pd.DataFrame([default_applicant()])
    s, base = explain_single(df, "stack_lr (Champion, calibrated OOF)")
    assert np.isfinite(base)
    assert len(s) >= 40
    assert np.all(np.isfinite(s.values))


def test_app_schema_validation():
    df = pd.DataFrame([default_applicant()]).drop(columns=["fico_range_low"])
    with pytest.raises(ValueError, match="Missing"):
        validate_and_prepare(df)


def test_fastapi_endpoints():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    r2 = client.post("/predict", json={"cost_ratio": 10.0, "records": [default_applicant()]})
    assert r2.status_code == 200
    body = r2.json()
    assert len(body["predictions"]) == 1
    assert body["predictions"][0]["decision"] in ("APPROVE", "DECLINE")
