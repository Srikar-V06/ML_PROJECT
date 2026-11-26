"""Block 3 (P3) core scaffolding: train-tail early stopping, CatBoost native
preprocessing, model factories, Optuna tuning/recovery, dual artifact export,
and single-run Test-lock guard.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from time import perf_counter

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

from src.audit.schema import CLEAN
from src.features.block2 import UnderwritingFeatures, build_preprocessor
from src.models.registry import get_model, register_model
from src.utils.io import artifact, digest, write_json

P3_BOOSTERS = (
    "lgbm_native",
    "lgbm_native_flags",
    "lgbm_imputed_flags",
    "xgb_hist",
    "hgb",
    "catboost_native",
    "gb_classic_50k",
    "adaboost_p3",
    "lgbm_monotone",
)

# Financial-logic signs on raw/engineered numeric fields (applied post-preprocessing by name).
DEFAULT_MONOTONE_SIGNS = {
    "dti": 1,
    "int_rate": 1,
    "revol_util": 1,
    "loan_to_income": 1,
    "payment_to_income": 1,
    "fico_range_low": -1,
    "annual_inc": -1,
}

P3_TUNE_FAMILIES = (
    "lightgbm",
    "xgboost",
    "catboost",
    "hgb",
    "gradient_boosting",
    "adaboost",
)

FAMILY_TO_FACTORY = {
    "lightgbm": "lgbm_native_flags",
    "xgboost": "xgb_hist",
    "catboost": "catboost_native",
    "hgb": "hgb",
    "gradient_boosting": "gb_classic_50k",
    "adaboost": "adaboost_p3",
}

DEFAULT_BOOSTER_PARAMS = {
    "lightgbm": {"learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20},
    "xgboost": {"learning_rate": 0.05, "max_depth": 6, "min_child_weight": 1.0},
    "catboost": {"learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 3.0},
    "hgb": {"learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 20},
    "gradient_boosting": {"learning_rate": 0.05, "max_depth": 3, "min_samples_leaf": 20},
    "adaboost": {"base_max_depth": 1, "learning_rate": 0.5},
}


# ---------------------------------------------------------------------------
# Preprocessing & Feature Helpers
# ---------------------------------------------------------------------------

def extract_keep_flags(missing_table, keep_cols):
    """Return missing-indicator feature names intersected with the 50 keep fields."""
    if isinstance(missing_table, (str, Path)):
        missing_table = pd.read_csv(missing_table)
    keep_set = set(keep_cols)
    mask = missing_table["add_missing_indicator"].astype(bool) & missing_table["feature_name"].isin(keep_set)
    return missing_table.loc[mask, "feature_name"].tolist()


def fold_missing_flags(X_fold):
    """Compute P1's >0.5% and <100% missingness rule inside a CV fitting fold."""
    missing = X_fold.isna().mean()
    return missing.index[(missing > 0.005) & (missing < 1.0)].tolist()


class CatBoostNativePreprocessor(BaseEstimator, TransformerMixin):
    """Preprocessor that preserves native categorical columns for CatBoost
    while applying P2's UnderwritingFeatures and optional missing flags.
    """

    def __init__(self, flags=(), engineered=True):
        self.flags = tuple(flags)
        self.engineered = engineered

    def fit(self, X, y=None):
        self.underwriting_ = UnderwritingFeatures(self.engineered).fit(X, y)
        Xt = self.underwriting_.transform(X)
        self.num_cols_ = Xt.select_dtypes(include="number").columns.tolist()
        self.cat_cols_ = [c for c in Xt.columns if c not in self.num_cols_]
        self.flag_cols_ = [f for f in self.flags if f in Xt.columns]
        self.feature_names_out_ = np.asarray(
            [*self.num_cols_, *self.cat_cols_, *(f"missing__{f}" for f in self.flag_cols_)],
            dtype=object,
        )
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_out_")
        Xt = self.underwriting_.transform(X)
        cols_dict = {}
        for col in self.num_cols_:
            cols_dict[col] = pd.to_numeric(Xt[col], errors="coerce").astype(float)
        for col in self.cat_cols_:
            s = Xt[col].astype(object)
            cols_dict[col] = s.where(s.notna(), "__missing__").astype(str)
        for f in self.flag_cols_:
            cols_dict[f"missing__{f}"] = Xt[f].isna().astype(float)
        return pd.DataFrame(cols_dict, index=Xt.index)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_


def stratified_train_subsample(X, y, issue_d=None, n_rows=50000, seed=42, out_path=None):
    """Deterministically subsample up to n_rows stratified by target x issue quarter."""
    if len(X) <= n_rows:
        ids = pd.Index(X.index, name="loan_id")
    else:
        if issue_d is not None:
            quarters = pd.to_datetime(issue_d).dt.to_period("Q").astype(str)
            strata = (y.astype(str) + "_" + quarters.astype(str)).copy()
        else:
            strata = y.astype(str).copy()
        # Merge rare strata (< 5 rows) into the largest stratum of the same target class.
        counts = strata.value_counts()
        rare = set(counts[counts < 5].index)
        if rare:
            for cls in y.unique():
                cls_mask = (y == cls)
                cls_counts = strata.loc[cls_mask].value_counts()
                common = cls_counts[cls_counts >= 5]
                fallback = common.index[0] if len(common) else f"__pool_{cls}"
                strata.loc[cls_mask & strata.isin(rare)] = fallback
        _, sub_idx = train_test_split(
            np.arange(len(X)),
            test_size=n_rows,
            random_state=seed,
            stratify=strata,
        )
        sub_idx = np.sort(sub_idx)
        ids = pd.Index(X.index[sub_idx], name="loan_id")

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"loan_id": ids.to_numpy()}).to_parquet(out_path, index=False)
    return ids, int(len(ids))


# ---------------------------------------------------------------------------
# Model Factories
# ---------------------------------------------------------------------------

def _lgbm_estimator(seed, **kwargs):
    from lightgbm import LGBMClassifier
    params = dict(random_state=seed, n_jobs=2, verbosity=-1, deterministic=True, force_col_wise=True)
    params.update(kwargs)
    return LGBMClassifier(**params)


def _xgb_estimator(seed, **kwargs):
    from xgboost import XGBClassifier
    params = dict(random_state=seed, tree_method="hist", n_jobs=2, eval_metric="auc")
    params.update(kwargs)
    return XGBClassifier(**params)


def _catboost_estimator(seed, cat_features=(), **kwargs):
    from catboost import CatBoostClassifier
    params = dict(
        random_seed=seed,
        thread_count=2,
        verbose=False,
        allow_writing_files=False,
        cat_features=list(cat_features),
    )
    params.update(kwargs)
    return CatBoostClassifier(**params)


@register_model("lgbm_native")
def lgbm_native_factory(X, flags, cfg):
    params = cfg.get("model_params", {}).get("lgbm_native", {})
    prep = build_preprocessor("tree", flags=(), impute_tree=False)
    return Pipeline([("preprocess", prep), ("model", _lgbm_estimator(cfg["seed"], **params))])


@register_model("lgbm_native_flags")
def lgbm_native_flags_factory(X, flags, cfg):
    params = cfg.get("model_params", {}).get("lgbm_native_flags", {})
    prep = build_preprocessor("tree", flags=flags, impute_tree=False)
    return Pipeline([("preprocess", prep), ("model", _lgbm_estimator(cfg["seed"], **params))])


@register_model("lgbm_imputed_flags")
def lgbm_imputed_flags_factory(X, flags, cfg):
    params = cfg.get("model_params", {}).get("lgbm_imputed_flags", {})
    prep = build_preprocessor("tree", flags=flags, impute_tree=True)
    return Pipeline([("preprocess", prep), ("model", _lgbm_estimator(cfg["seed"], **params))])


@register_model("xgb_hist")
def xgb_hist_factory(X, flags, cfg):
    params = cfg.get("model_params", {}).get("xgb_hist", {})
    prep = build_preprocessor("tree", flags=flags, impute_tree=False)
    return Pipeline([("preprocess", prep), ("model", _xgb_estimator(cfg["seed"], **params))])


@register_model("hgb")
def hgb_factory(X, flags, cfg):
    params = dict(cfg.get("model_params", {}).get("hgb", {}))
    params["early_stopping"] = False
    prep = build_preprocessor("tree", flags=flags, impute_tree=False)
    return Pipeline([("preprocess", prep), ("model", HistGradientBoostingClassifier(random_state=cfg["seed"], **params))])


@register_model("catboost_native")
def catboost_native_factory(X, flags, cfg):
    params = cfg.get("model_params", {}).get("catboost_native", {})
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    prep = CatBoostNativePreprocessor(flags=flags, engineered=True)
    return Pipeline([("preprocess", prep), ("model", _catboost_estimator(cfg["seed"], cat_features=cat_cols, **params))])


@register_model("gb_classic_50k")
def gb_classic_50k_factory(X, flags, cfg):
    params = cfg.get("model_params", {}).get("gb_classic_50k", {})
    prep = build_preprocessor("tree", flags=flags, impute_tree=True)
    return Pipeline([("preprocess", prep), ("model", GradientBoostingClassifier(random_state=cfg["seed"], **params))])


@register_model("adaboost_p3")
def adaboost_p3_factory(X, flags, cfg):
    params = dict(cfg.get("model_params", {}).get("adaboost_p3", {}))
    base_depth = int(params.pop("base_max_depth", 1))
    base = DecisionTreeClassifier(max_depth=base_depth, random_state=cfg["seed"])
    prep = build_preprocessor("tree", flags=flags, impute_tree=True)
    return Pipeline([
        ("preprocess", prep),
        ("model", AdaBoostClassifier(estimator=base, random_state=cfg["seed"], **params)),
    ])


def monotone_vector(feature_names, signs):
    """Map transformed column names (e.g. 'numeric__dti') to LightGBM monotone constraints."""
    out = []
    for name in feature_names:
        name = str(name)
        base = name.split("__", 1)[1] if name.startswith("numeric__") else None
        out.append(int(signs.get(base, 0)) if base is not None else 0)
    return out


@register_model("lgbm_monotone")
def lgbm_monotone_factory(X, flags, cfg):
    """Native-NaN LightGBM (no flags, same inputs as lgbm_native) with monotone constraints.

    Constraint positions are resolved from the preprocessor fitted on the same X the
    pipeline will be fitted on, so names and positions always agree.
    """
    params = dict(cfg.get("model_params", {}).get("lgbm_monotone", {}))
    signs = cfg.get("monotone_signs", DEFAULT_MONOTONE_SIGNS)
    names = build_preprocessor("tree", flags=(), impute_tree=False).fit(X).get_feature_names_out()
    constraints = monotone_vector(names, signs)
    prep = build_preprocessor("tree", flags=(), impute_tree=False)
    est = _lgbm_estimator(
        cfg["seed"],
        monotone_constraints=constraints,
        monotone_constraints_method="advanced",
        **params,
    )
    return Pipeline([("preprocess", prep), ("model", est)])


# ---------------------------------------------------------------------------
# Train-Tail Early-Stopping Helper
# ---------------------------------------------------------------------------

def chronological_train_tail_split(X, y, issue_d=None, tail_fraction=0.10):
    """Split Train chronologically into first (1 - tail_fraction) and last tail_fraction."""
    if not (0.0 < tail_fraction < 0.5):
        raise ValueError("tail_fraction must be in (0, 0.5)")
    if issue_d is not None:
        order = np.lexsort((np.asarray(X.index), pd.to_datetime(issue_d).to_numpy()))
        ordered_index = X.index[order]
    else:
        ordered_index = X.index
    n = len(ordered_index)
    cut = int(round(n * (1.0 - tail_fraction)))
    cut = max(1, min(n - 1, cut))
    fit_idx = ordered_index[:cut]
    tail_idx = ordered_index[cut:]
    return fit_idx, tail_idx


def _round_param_name(est):
    cls_name = type(est).__name__
    if cls_name == "CatBoostClassifier":
        return "iterations"
    if cls_name == "HistGradientBoostingClassifier":
        return "max_iter"
    return "n_estimators"


def _fit_estimator_with_early_stop(
    est,
    X_fit_t,
    y_fit,
    X_tail_t,
    y_tail,
    patience=50,
    max_rounds=2000,
    adaboost_max_rounds=500,
):
    """Fit an estimator on already-preprocessed matrices and return best_iteration."""
    cls_name = type(est).__name__

    if cls_name == "LGBMClassifier":
        import lightgbm as lgb
        est.set_params(n_estimators=max_rounds)
        est.fit(
            X_fit_t,
            y_fit,
            eval_set=[(X_tail_t, y_tail)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(patience, verbose=False)],
        )
        best_iter = int(getattr(est, "best_iteration_", None) or max_rounds)
    elif cls_name == "XGBClassifier":
        est.set_params(n_estimators=max_rounds, early_stopping_rounds=patience, eval_metric="auc")
        est.fit(X_fit_t, y_fit, eval_set=[(X_tail_t, y_tail)], verbose=False)
        best_raw = getattr(est, "best_iteration", None)
        best_iter = int(best_raw + 1) if best_raw is not None else max_rounds
    elif cls_name == "CatBoostClassifier":
        est.set_params(iterations=max_rounds, early_stopping_rounds=patience, eval_metric="AUC")
        est.fit(X_fit_t, y_fit, eval_set=(X_tail_t, y_tail), use_best_model=True, verbose=False)
        best_raw = est.get_best_iteration()
        best_iter = int(best_raw + 1) if best_raw is not None and best_raw >= 0 else max_rounds
    elif cls_name in {"HistGradientBoostingClassifier", "AdaBoostClassifier", "GradientBoostingClassifier"}:
        cap = adaboost_max_rounds if cls_name == "AdaBoostClassifier" else max_rounds
        round_key = _round_param_name(est)
        extra = {"early_stopping": False} if cls_name == "HistGradientBoostingClassifier" else {}
        est.set_params(**{round_key: cap, **extra})
        est.fit(X_fit_t, y_fit)
        best_iter, best_auc = 1, -1.0
        for step, proba in enumerate(est.staged_predict_proba(X_tail_t), start=1):
            score = float(roc_auc_score(y_tail, proba[:, 1]))
            if score > best_auc:
                best_auc = score
                best_iter = step
    else:
        raise TypeError(f"Unsupported estimator for early stopping: {cls_name}")

    return max(1, int(best_iter))


def _fit_with_early_stop(pipe, X_fit, y_fit, X_tail, y_tail, patience=50, max_rounds=2000, adaboost_max_rounds=500):
    """Fit a pipeline on (X_fit, y_fit) and determine best_iteration on (X_tail, y_tail)."""
    prep = pipe.named_steps["preprocess"]
    est = pipe.named_steps["model"]
    X_fit_t = prep.fit_transform(X_fit, y_fit)
    X_tail_t = prep.transform(X_tail)
    return _fit_estimator_with_early_stop(
        est,
        X_fit_t,
        y_fit,
        X_tail_t,
        y_tail,
        patience=patience,
        max_rounds=max_rounds,
        adaboost_max_rounds=adaboost_max_rounds,
    )


def fit_with_train_tail_early_stopping(
    pipeline_factory,
    X,
    y,
    flags,
    cfg,
    *,
    issue_d=None,
    tail_fraction=0.10,
    patience=50,
    max_rounds=2000,
    adaboost_max_rounds=500,
):
    """Determine best_iteration on chronological Train tail (never Val), then refit on 100% Train."""
    fit_idx, tail_idx = chronological_train_tail_split(X, y, issue_d=issue_d, tail_fraction=tail_fraction)
    X_fit, y_fit = X.loc[fit_idx], y.loc[fit_idx]
    X_tail, y_tail = X.loc[tail_idx], y.loc[tail_idx]

    probe_pipe = pipeline_factory(X_fit, flags, cfg)
    best_iter = _fit_with_early_stop(
        probe_pipe,
        X_fit,
        y_fit,
        X_tail,
        y_tail,
        patience=patience,
        max_rounds=max_rounds,
        adaboost_max_rounds=adaboost_max_rounds,
    )
    scaled_rounds = max(1, int(round(best_iter / (1.0 - tail_fraction))))

    final_pipe = pipeline_factory(X, flags, cfg)
    est = final_pipe.named_steps["model"]
    round_key = _round_param_name(est)
    clear_es = {}
    if type(est).__name__ == "XGBClassifier":
        clear_es["early_stopping_rounds"] = None
    elif type(est).__name__ == "HistGradientBoostingClassifier":
        clear_es["early_stopping"] = False
    est.set_params(**{round_key: scaled_rounds, **clear_es})
    final_pipe.fit(X, y)

    meta = {
        "best_iteration": int(best_iter),
        "refit_rounds": int(scaled_rounds),
        "tail_fraction": float(tail_fraction),
        "fit_rows": int(len(X)),
        "probe_fit_rows": int(len(X_fit)),
        "probe_tail_rows": int(len(X_tail)),
    }
    return final_pipe, meta


# ---------------------------------------------------------------------------
# Optuna Harness & P2 Warm-Start Recovery
# ---------------------------------------------------------------------------

def p3_search_space(trial, family):
    if family == "lightgbm":
        return {
            "num_leaves": trial.suggest_int("num_leaves", 8, 128),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": 1,
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        }
    if family == "xgboost":
        return {
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        }
    if family == "catboost":
        return {
            "depth": trial.suggest_int("depth", 2, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 20.0, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        }
    if family == "hgb":
        return {
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 200),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        }
    if family == "gradient_boosting":
        return {
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 200),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        }
    if family == "adaboost":
        return {
            "base_max_depth": trial.suggest_int("base_max_depth", 1, 3),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 1.0, log=True),
        }
    raise KeyError(f"Unknown tuning family: {family}")


def load_p2_warm_start(family, cfg):
    """Load P2 search JSON params mapped into P3's search space (excluding round counts)."""
    p2_path = artifact(cfg, f"block2/search_{family}.json")
    if not p2_path.exists():
        return None
    raw = json.loads(p2_path.read_text(encoding="utf-8")).get("params", {})
    filtered = {}
    for k, v in raw.items():
        if k in {"n_estimators", "iterations"}:
            continue
        filtered[k] = v
    if family == "adaboost" and "base_max_depth" not in filtered:
        filtered["base_max_depth"] = 1
    return filtered or None


def tune_booster_p3(
    family,
    X,
    y,
    cfg,
    *,
    trials=30,
    timeout_seconds=1500,
    fit_row_limit=25000,
    max_rounds=2000,
    adaboost_max_rounds=500,
    patience=50,
    p3_root=None,
):
    """Tune a booster family over the 4 frozen expanding Train folds using Optuna."""
    t0 = perf_counter()
    splits_root = Path(cfg["paths"]["splits"])
    membership = pd.read_parquet(splits_root / "train_expanding_cv.parquet")
    folds = []
    for _, group in membership.groupby("fold", sort=True):
        fit_ids = group.loc[group.role.eq("train"), "id"]
        val_ids = group.loc[group.role.eq("val"), "id"]
        if fit_row_limit and len(fit_ids) > fit_row_limit:
            fit_ids = fit_ids.sample(n=fit_row_limit, random_state=cfg["seed"])
        fit_x = X.loc[fit_ids]
        folds.append((fit_ids, val_ids, fold_missing_flags(fit_x)))

    factory_name = FAMILY_TO_FACTORY[family]
    factory = get_model(factory_name)

    # Pre-fit each expanding fold's preprocessor ONCE strictly on fold fit_ids (zero leakage).
    prepared_folds = []
    for fit_ids, val_ids, flags in folds:
        X_f, y_f = X.loc[fit_ids], y.loc[fit_ids]
        X_v, y_v = X.loc[val_ids], y.loc[val_ids]
        base_pipe = factory(X_f, flags, cfg)
        prep = base_pipe.named_steps["preprocess"]
        X_f_t = prep.fit_transform(X_f, y_f)
        X_v_t = prep.transform(X_v)
        prepared_folds.append((X_f, X_f_t, y_f, X_v_t, y_v, flags))

    def objective(trial):
        params = dict(p3_search_space(trial, family))
        scores = []
        best_iters = []
        for X_f, X_f_t, y_f, X_v_t, y_v, flags in prepared_folds:
            trial_cfg = {**cfg, "model_params": {**cfg.get("model_params", {}), factory_name: dict(params)}}
            est = factory(X_f, flags, trial_cfg).named_steps["model"]
            best_iter = _fit_estimator_with_early_stop(
                est,
                X_f_t,
                y_f,
                X_v_t,
                y_v,
                patience=patience,
                max_rounds=max_rounds,
                adaboost_max_rounds=adaboost_max_rounds,
            )
            cls_name = type(est).__name__
            if cls_name in {"HistGradientBoostingClassifier", "AdaBoostClassifier", "GradientBoostingClassifier"}:
                preds = None
                for step, proba in enumerate(est.staged_predict_proba(X_v_t), start=1):
                    if step == best_iter:
                        preds = proba[:, 1]
                        break
            else:
                preds = est.predict_proba(X_v_t)[:, 1]
            scores.append(float(roc_auc_score(y_v, preds)))
            best_iters.append(int(best_iter))
        trial.set_user_attr("fold_auc", scores)
        trial.set_user_attr("fold_best_iter", best_iters)
        return float(np.mean(scores))

    p3_dir = Path(p3_root) if p3_root is not None else Path("artifacts/p3")
    p3_dir.mkdir(parents=True, exist_ok=True)
    db_path = (p3_dir / f"search_{family}.sqlite3").resolve().as_posix().replace("{", "{{").replace("}", "}}")
    study_name = f"p3-{family}-{digest(splits_root / 'manifest.json')[:12]}-{cfg['seed']}-{fit_row_limit}"
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{db_path}",
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=cfg["seed"]),
    )

    warm_start = load_p2_warm_start(family, cfg)
    if len(study.trials) == 0 and warm_start is not None:
        study.enqueue_trial(warm_start)

    completed_before = len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)))
    if completed_before < trials:
        study.optimize(objective, n_trials=trials - completed_before, timeout=timeout_seconds)

    completed_trials = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
    elapsed = perf_counter() - t0
    if completed_trials:
        best = study.best_trial
        params = dict(best.params)
        best_auc = float(best.value)
        status = "complete" if len(completed_trials) >= trials else "partial_timeout"
        source = "optuna_p3"
    else:
        params = dict(DEFAULT_BOOSTER_PARAMS[family])
        best_auc = None
        status = "fallback_default"
        source = "default_params"

    record = {
        "family": family,
        "status": status,
        "completed_trials": len(completed_trials),
        "warm_started": warm_start is not None,
        "source": source,
        "best_mean_auc": best_auc,
        "params_json": json.dumps(params, sort_keys=True),
        "elapsed_seconds": round(elapsed, 2),
    }
    write_json(
        p3_dir / "runs" / f"tune_{family}.json",
        {**record, "params": params, "fit_row_limit": fit_row_limit},
    )
    rebuild_recovery_table(p3_dir)
    return params, record


def rebuild_recovery_table(p3_root="artifacts/p3"):
    """Aggregate all per-family tune_<family>.json records into recovery.csv."""
    p3_dir = Path(p3_root)
    runs_dir = p3_dir / "runs"
    cols = [
        "family",
        "status",
        "completed_trials",
        "warm_started",
        "source",
        "best_mean_auc",
        "params_json",
        "elapsed_seconds",
    ]
    rows = []
    if runs_dir.exists():
        for path in sorted(runs_dir.glob("tune_*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            rows.append({c: data.get(c) for c in cols})
    out_df = pd.DataFrame(rows, columns=cols)
    recovery_path = p3_dir / "tables" / "recovery.csv"
    recovery_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(recovery_path, index=False)
    return out_df


# ---------------------------------------------------------------------------
# Artifact Writers & Dual Exporter
# ---------------------------------------------------------------------------

def to_brief_predictions(pred_df):
    """Convert canonical (loan_id, y_true, y_prob, split) to brief (id, y_true, p_default)."""
    return pd.DataFrame({
        "id": pred_df["loan_id"].astype("int64"),
        "y_true": pred_df["y_true"].astype("int8"),
        "p_default": pred_df["y_prob"].astype("float64"),
    })


def save_p3_artifacts(
    name,
    model,
    pred_df,
    split,
    *,
    run_meta=None,
    p3_root="artifacts/p3",
    brief_root=".",
    timestamp=None,
):
    """Save model, predictions, and run metadata to both artifacts/p3/ and brief layout."""
    p3_dir = Path(p3_root)
    b_dir = Path(brief_root)
    ts = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # 1. Model joblibs
    p3_model_path = p3_dir / "models" / f"{name}.joblib"
    p3_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, p3_model_path, compress=3)

    brief_model_path = b_dir / "models" / f"{name}.joblib"
    brief_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, brief_model_path, compress=3)

    # 2. Predictions (canonical + brief)
    p3_pred_path = p3_dir / "predictions" / f"{name}_{split}.parquet"
    p3_pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(p3_pred_path, index=False)

    brief_pred_path = b_dir / "reports" / "preds" / f"{name}_{split}.parquet"
    brief_pred_path.parent.mkdir(parents=True, exist_ok=True)
    to_brief_predictions(pred_df).to_parquet(brief_pred_path, index=False)

    # 3. Run metadata
    if run_meta is not None:
        meta = dict(run_meta, model=name, split=split, timestamp=ts)
        write_json(p3_dir / "runs" / f"{ts}_{name}.json", meta)
        write_json(b_dir / "reports" / "runs" / f"{ts}_{name}.json", meta)

    return {
        "p3_model": p3_model_path,
        "brief_model": brief_model_path,
        "p3_pred": p3_pred_path,
        "brief_pred": brief_pred_path,
    }


def save_benchmark_table(df, *, p3_root="artifacts/p3", brief_root="."):
    """Write benchmark.csv to both artifacts/p3/tables/ and reports/tables/."""
    p1 = Path(p3_root) / "tables" / "benchmark.csv"
    p2 = Path(brief_root) / "reports" / "tables" / "benchmark.csv"
    for p in (p1, p2):
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
    return p1, p2


# ---------------------------------------------------------------------------
# Single-Run Test-Lock Guard
# ---------------------------------------------------------------------------

def write_model_manifest(models_map, feature_order, manifest_path, *, champion=None, val_thresholds=None):
    """Create model_manifest.json with SHA-256 digests and exact feature column order."""
    entries = {}
    for name, path in sorted(models_map.items()):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Missing model file for manifest: {p}")
        entries[name] = {
            "path": str(p),
            "sha256": digest(p),
            "bytes": int(p.stat().st_size),
        }
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_order": list(feature_order),
        "champion": champion,
        "val_thresholds": val_thresholds or {},
        "models": entries,
    }
    write_json(manifest_path, manifest)
    return manifest


def verify_manifest_and_acquire_test_lock(manifest_path, lock_path, expected_feature_order=None):
    """Refuse execution if lock_path exists; verify all model SHA-256s and feature order; write lock."""
    lock = Path(lock_path)
    if lock.exists():
        raise RuntimeError(
            f"Test lock already exists at {lock}. Single final Test evaluation has already run."
        )
    m_path = Path(manifest_path)
    if not m_path.exists():
        raise FileNotFoundError(f"Model manifest not found: {m_path}")
    manifest = json.loads(m_path.read_text(encoding="utf-8"))

    if expected_feature_order is not None and list(manifest.get("feature_order", [])) != list(expected_feature_order):
        raise ValueError("Feature column order mismatch against model_manifest.json")

    for name, info in manifest.get("models", {}).items():
        m_file = Path(info["path"])
        if not m_file.exists():
            raise FileNotFoundError(f"Manifest model missing: {name} -> {m_file}")
        actual = digest(m_file)
        if actual != info["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {name}: expected {info['sha256']}, got {actual}")

    sha = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], capture_output=True, text=True)
    lock_payload = {
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": sha.stdout.strip() or None,
        "manifest_path": str(m_path),
        "manifest_sha256": digest(m_path),
        "model_count": len(manifest.get("models", {})),
    }
    write_json(lock, lock_payload)
    return manifest
