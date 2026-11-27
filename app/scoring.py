"""Shared pipeline loading, schema validation, scoring, and SHAP explanation helpers for UI and API."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.analysis.block4_shap import aggregate, explain
from src.features.block2 import FEATURE_GROUPS
from src.models.baselines import model_frame

P3 = Path("artifacts/p3")
P4 = Path("artifacts/p4")
P5 = Path("artifacts/p5")
TEMPLATE_PATH = Path("app/template_row.json")

UI_MODELS = {
    "stack_lr (Champion, calibrated OOF)": ("stack_lr", None),
    "catboost (Platt-calibrated)": ("catboost", P4 / "calibration" / "catboost_platt.joblib"),
    "xgb_hist (Platt-calibrated)": ("xgb_hist", P4 / "calibration" / "xgb_hist_platt.joblib"),
    "lgbm_native_flags (Platt-calibrated)": ("lgbm_native_flags", P4 / "calibration" / "lgbm_native_flags_platt.joblib"),
    "lgbm_monotone (Monotone Booster)": ("lgbm_monotone", None),
    "logreg_baseline (Linear Baseline)": ("logreg_baseline", None),
}


def load_manifest() -> dict:
    return json.loads((P3 / "model_manifest.json").read_text())


def feature_order() -> list[str]:
    return load_manifest()["feature_order"]


def load_pipeline(model_key: str, apply_platt: bool = True):
    manifest = load_manifest()
    base_name, cal_path = UI_MODELS[model_key]
    pipe = joblib.load(manifest["models"][base_name]["path"])
    cal = joblib.load(cal_path) if (apply_platt and cal_path is not None and cal_path.exists()) else None
    return base_name, pipe, cal


def default_applicant() -> dict:
    if TEMPLATE_PATH.exists():
        return json.loads(TEMPLATE_PATH.read_text())
    return {
        "loan_amnt": 14000.0, "term": 36, "int_rate": 12.99, "installment": 471.70,
        "grade": "C", "sub_grade": "C1", "emp_title": "Manager", "emp_length": 5.0,
        "home_ownership": "MORTGAGE", "annual_inc": 68000.0, "verification_status": "Verified",
        "purpose": "debt_consolidation", "addr_state": "CA", "dti": 17.5,
        "delinq_2yrs": 0.0, "earliest_cr_line": "2002-06-01", "fico_range_low": 695.0,
        "inq_last_6mths": 0.0, "mths_since_last_delinq": np.nan, "mths_since_last_record": np.nan,
        "open_acc": 11.0, "pub_rec": 0.0, "revol_bal": 12500.0, "revol_util": 52.0,
        "total_acc": 24.0, "initial_list_status": "w", "collections_12_mths_ex_med": 0.0,
        "mths_since_last_major_derog": np.nan, "application_type": "Individual",
        "annual_inc_joint": np.nan, "dti_joint": np.nan, "verification_status_joint": None,
        "acc_now_delinq": 0.0, "tot_coll_amt": 0.0, "tot_cur_bal": 95000.0,
        "open_acc_6m": np.nan, "open_act_il": np.nan, "open_il_12m": np.nan, "open_il_24m": np.nan,
        "mths_since_rcnt_il": np.nan, "total_bal_il": np.nan, "il_util": np.nan,
        "open_rv_12m": np.nan, "open_rv_24m": np.nan, "max_bal_bc": np.nan, "all_util": np.nan,
        "total_rev_hi_lim": 25000.0, "inq_fi": np.nan, "total_cu_tl": np.nan, "inq_last_12m": np.nan,
    }


def validate_and_prepare(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    cols = feature_order()
    missing = [c for c in cols if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing {len(missing)} required columns: {missing[:8]}{'...' if len(missing) > 8 else ''}")
    df = df_raw[cols].copy()
    if "earliest_cr_line" in df.columns:
        df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], errors="coerce")
    return model_frame(df), []


def score_dataframe(df_raw: pd.DataFrame, model_key: str, apply_platt: bool = True) -> np.ndarray:
    X, _ = validate_and_prepare(df_raw)
    _, pipe, cal = load_pipeline(model_key, apply_platt=apply_platt)
    p = pipe.predict_proba(X)[:, 1]
    if cal is not None:
        p = cal.transform(p)
    return np.asarray(p, dtype=float)


def explain_single(row_raw: pd.DataFrame, model_key: str) -> tuple[pd.Series, float]:
    """Return (field-level SHAP Series sorted by |SHAP| desc, base_logit) for 1 applicant."""
    X, _ = validate_and_prepare(row_raw)
    base_name, pipe, _ = load_pipeline(model_key, apply_platt=False)
    fields = feature_order() + list(FEATURE_GROUPS["engineered"])

    if base_name == "stack_lr":
        meta = json.loads((P3 / "stack" / "stack_lr_meta.json").read_text())
        weights = meta["coefficients"]
        phi_total = None
        base_total = float(meta["intercept"])
        for m_name, m_pipe in pipe.members:
            w = float(weights[m_name])
            if m_name == "logreg_baseline":
                # StandardScaler already centers to train mean; for 1 row Z.mean(0)==Z so use coef*Z directly
                pre, mod = m_pipe[:-1], m_pipe[-1]
                Z = pre.transform(X)
                Z = np.asarray(Z.toarray() if hasattr(Z, "toarray") else Z, dtype=float)
                from src.analysis.block4_shap import aggregate
                phi_m = aggregate(mod.coef_[0] * Z, list(pre.get_feature_names_out()), fields)
                b_m = float(mod.intercept_[0])
            else:
                phi_m, b_m = explain(m_pipe, X, fields)
            phi_total = phi_m * w if phi_total is None else phi_total.add(phi_m * w, fill_value=0.0)
            base_total += w * b_m
        s = phi_total.iloc[0]
    else:
        phi, base_total = explain(pipe, X, fields)
        s = phi.iloc[0]
    return s.reindex(s.abs().sort_values(ascending=False).index), float(base_total)
