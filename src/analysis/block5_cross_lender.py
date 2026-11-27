"""Block 5 — Cross-lender generalization study (Lending Club -> Prosper Loan Data)."""
from pathlib import Path
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data.splits import load  # noqa: E402
from src.evaluation.calibration import Platt, ece  # noqa: E402
from src.utils.io import config  # noqa: E402

P5 = Path("artifacts/p5")
PROSPER_GRADE_MAP = {"AA": 1, "A": 2, "B": 3, "C": 4, "D": 5, "E": 6, "HR": 7, "NC": 7}
LC_GRADE_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}

COMMON_NUM = [
    "loan_amnt", "term", "int_rate", "annual_inc", "dti", "fico_range_low",
    "inq_last_6mths", "open_acc", "acc_now_delinq", "delinq_amnt",
    "revol_bal", "revol_util", "emp_length", "credit_hist_months", "loan_to_income",
]
COMMON_CAT = ["home_ownership", "addr_state"]


def _align_frame(df: pd.DataFrame, is_prosper: bool, include_grade: bool = False) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in ["loan_amnt", "term", "int_rate", "annual_inc", "dti", "fico_range_low",
              "inq_last_6mths", "open_acc", "acc_now_delinq", "delinq_amnt",
              "revol_bal", "revol_util", "emp_length"]:
        out[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    issue = pd.to_datetime(df["issue_d"], errors="coerce")
    earliest = pd.to_datetime(df["earliest_cr_line"], errors="coerce")
    out["credit_hist_months"] = ((issue - earliest).dt.days / 30.4375).clip(lower=0.0)
    inc = out["annual_inc"].replace(0, np.nan)
    out["loan_to_income"] = (out["loan_amnt"] / inc).clip(0.0, 5.0)

    if is_prosper:
        out["home_ownership"] = df["home_ownership"].astype(str).fillna("NON_OWNER")
    else:
        ho = df["home_ownership"].astype(str)
        out["home_ownership"] = np.where(ho.isin(["OWN", "MORTGAGE"]), "OWNER", "NON_OWNER")

    out["addr_state"] = df["addr_state"].astype(str).fillna("MISSING")
    if include_grade:
        gmap = PROSPER_GRADE_MAP if is_prosper else LC_GRADE_MAP
        out["grade_ord"] = df["grade"].astype(str).map(gmap).astype(float)
    return out


def _build_pipeline(model_name: str, num_cols: list[str], cat_cols: list[str], seed: int = 42) -> Pipeline:
    if model_name == "logreg":
        num_pipe = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())])
        cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                             ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
        pre = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])
        est = LogisticRegression(C=0.5, max_iter=500, random_state=seed)
    elif model_name == "lgbm":
        cat_pipe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        pre = ColumnTransformer([("num", "passthrough", num_cols), ("cat", cat_pipe, cat_cols)])
        est = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.04, num_leaves=31,
                                 subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbose=-1)
    elif model_name == "xgb":
        cat_pipe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        pre = ColumnTransformer([("num", "passthrough", num_cols), ("cat", cat_pipe, cat_cols)])
        est = xgb.XGBClassifier(n_estimators=500, learning_rate=0.04, max_depth=5,
                                subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                eval_metric="logloss", random_state=seed, n_jobs=-1)
    elif model_name == "catboost":
        cat_pipe = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="MISSING")),
                             ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
        pre = ColumnTransformer([("num", "passthrough", num_cols), ("cat", cat_pipe, cat_cols)])
        est = CatBoostClassifier(iterations=600, learning_rate=0.05, depth=6, random_seed=seed, verbose=0, thread_count=-1)
    else:
        raise ValueError(model_name)
    return Pipeline([("preprocess", pre), ("model", est)])


def _eval_row(model: str, feature_set: str, regime: str, y: np.ndarray, p: np.ndarray) -> dict:
    thr = 1.0 / 11.0
    rej = p >= thr
    fp = int(np.sum(rej & (y == 0)))
    fn = int(np.sum(~rej & (y == 1)))
    return {
        "model": model,
        "feature_set": feature_set,
        "regime": regime,
        "n": int(len(y)),
        "default_rate": float(y.mean()),
        "mean_pred_p": float(p.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": float(ece(y, p)),
        "approval_rate_10_to_1": float((~rej).mean()),
        "cost_per_1k_R10": float((fp + 10.0 * fn) * 1000.0 / len(y)),
    }


def run(cfg=None):
    cfg = cfg or config()
    (P5 / "tables").mkdir(parents=True, exist_ok=True)
    (P5 / "plots").mkdir(parents=True, exist_ok=True)

    lc_frames = load(cfg, names=("train", "test"))
    lc_train, lc_test = lc_frames["train"], lc_frames["test"]

    prosper = pd.read_parquet(cfg["paths"]["prosper_common"]).sort_values(["issue_d", "loan_id"]).reset_index(drop=True)
    cut = int(0.20 * len(prosper))
    p_calib = prosper.iloc[:cut].copy()
    p_test = prosper.iloc[cut:].copy()

    rows = []
    # 1. Zero-shot LC -> Prosper (common_no_grade) + Platt target calibration
    X_lc_tr = _align_frame(lc_train, is_prosper=False, include_grade=False)
    X_lc_te = _align_frame(lc_test, is_prosper=False, include_grade=False)
    X_pr_cal = _align_frame(p_calib, is_prosper=True, include_grade=False)
    X_pr_te = _align_frame(p_test, is_prosper=True, include_grade=False)

    y_lc_tr = lc_train["target"].to_numpy(dtype=int)
    y_lc_te = lc_test["target"].to_numpy(dtype=int)
    y_pr_cal = p_calib["target"].to_numpy(dtype=int)
    y_pr_te = p_test["target"].to_numpy(dtype=int)

    for m in ["logreg", "lgbm", "xgb", "catboost"]:
        pipe = _build_pipeline(m, COMMON_NUM, COMMON_CAT, seed=cfg["seed"])
        pipe.fit(X_lc_tr, y_lc_tr)
        rows.append(_eval_row(m, "common_16_no_grade", "LC_test_in_domain", y_lc_te, pipe.predict_proba(X_lc_te)[:, 1]))

        p_raw_cal = pipe.predict_proba(X_pr_cal)[:, 1]
        p_raw_te = pipe.predict_proba(X_pr_te)[:, 1]
        rows.append(_eval_row(m, "common_16_no_grade", "Prosper_zero_shot_raw", y_pr_te, p_raw_te))

        platt = Platt().fit(p_raw_cal, y_pr_cal)
        rows.append(_eval_row(m, "common_16_no_grade", "Prosper_zero_shot_platt", y_pr_te, platt.transform(p_raw_te)))

    # 2. Check with mapped grade for lgbm
    X_lc_tr_g = _align_frame(lc_train, is_prosper=False, include_grade=True)
    X_lc_te_g = _align_frame(lc_test, is_prosper=False, include_grade=True)
    X_pr_cal_g = _align_frame(p_calib, is_prosper=True, include_grade=True)
    X_pr_te_g = _align_frame(p_test, is_prosper=True, include_grade=True)

    pipe_g = _build_pipeline("lgbm", COMMON_NUM + ["grade_ord"], COMMON_CAT, seed=cfg["seed"])
    pipe_g.fit(X_lc_tr_g, y_lc_tr)
    rows.append(_eval_row("lgbm", "common_17_with_grade", "LC_test_in_domain", y_lc_te, pipe_g.predict_proba(X_lc_te_g)[:, 1]))
    p_g_raw = pipe_g.predict_proba(X_pr_te_g)[:, 1]
    rows.append(_eval_row("lgbm", "common_17_with_grade", "Prosper_zero_shot_raw", y_pr_te, p_g_raw))
    platt_g = Platt().fit(pipe_g.predict_proba(X_pr_cal_g)[:, 1], y_pr_cal)
    rows.append(_eval_row("lgbm", "common_17_with_grade", "Prosper_zero_shot_platt", y_pr_te, platt_g.transform(p_g_raw)))

    # 3. Target-domain adaptation: refit / fine-tune on 1k, 3k, and full prosper_calib (8.8k)
    for n_ft in [1000, 3000, len(p_calib)]:
        sub_X = X_pr_cal.iloc[-n_ft:]
        sub_y = y_pr_cal[-n_ft:]
        for m in ["lgbm", "logreg"]:
            p_ft = _build_pipeline(m, COMMON_NUM, COMMON_CAT, seed=cfg["seed"])
            p_ft.fit(sub_X, sub_y)
            rows.append(_eval_row(m, "common_16_no_grade", f"Prosper_refit_{n_ft}", y_pr_te, p_ft.predict_proba(X_pr_te)[:, 1]))

    df = pd.DataFrame(rows)
    df.to_csv(P5 / "tables" / "cross_lender.csv", index=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    sub = df[df["feature_set"] == "common_16_no_grade"]
    regimes = ["LC_test_in_domain", "Prosper_zero_shot_raw", "Prosper_zero_shot_platt"]
    x = np.arange(len(regimes))
    w = 0.2
    for i, m in enumerate(["logreg", "lgbm", "xgb", "catboost"]):
        d = sub[(sub["model"] == m) & (sub["regime"].isin(regimes))].set_index("regime").loc[regimes]
        axes[0].bar(x + (i - 1.5) * w, d["roc_auc"], width=w, label=m)
        axes[1].bar(x + (i - 1.5) * w, d["ece"], width=w, label=m)
    axes[0].set_xticks(x, ["LC Test\n(In-Domain)", "Prosper Test\n(Zero-Shot Raw)", "Prosper Test\n(+Platt Calib)"])
    axes[1].set_xticks(x, ["LC Test\n(In-Domain)", "Prosper Test\n(Zero-Shot Raw)", "Prosper Test\n(+Platt Calib)"])
    axes[0].set(ylim=(0.55, 0.78), ylabel="ROC-AUC", title="Cross-Lender Discrimination (LC -> Prosper)")
    axes[1].set(ylabel="ECE (Calibration Error)", title="Calibration Shift & Target Platt Fix")
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(P5 / "plots" / "cross_lender.png", dpi=150)
    plt.close(fig)
    return df


if __name__ == "__main__":
    out = run()
    print(out[["model", "feature_set", "regime", "roc_auc", "pr_auc", "brier", "ece", "cost_per_1k_R10"]].to_string(index=False))
