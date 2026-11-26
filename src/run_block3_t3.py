"""Step T3 runner: Optuna tuning, chronological train-tail early stopping,
full-Train refits (50k stratified for classic GB), and Val-only scoring.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import pandas as pd
from threadpoolctl import threadpool_limits

from src.block3 import (
    extract_keep_flags,
    fit_with_train_tail_early_stopping,
    rebuild_recovery_table,
    save_p3_artifacts,
    stratified_train_subsample,
    tune_booster_p3,
)
from src.data.splits import load
from src.evaluation.metrics import evaluate
from src.models.baselines import model_frame
from src.models.registry import get_model
from src.utils.io import artifact, config, write_json
from src.utils.seed import set_seed

P3_ROOT = Path("artifacts/p3")


def _load_train_val(cfg):
    frames = load(cfg, names=("train", "val"))
    tr, va = frames["train"], frames["val"]
    audit = pd.read_csv(artifact(cfg, "feature_audit.csv"))
    cols = audit.loc[audit.decision.eq("keep"), "feature_name"].tolist()
    flags = extract_keep_flags(artifact(cfg, "missing_value_table.csv"), cols)
    X_tr = model_frame(tr[cols])
    y_tr = tr.target
    X_va = model_frame(va[cols])
    y_va = va.target
    return tr, va, cols, flags, X_tr, y_tr, X_va, y_va


def _fit_and_score_val(
    save_name,
    factory_name,
    family,
    params,
    X_tr,
    y_tr,
    issue_tr,
    X_va,
    y_va,
    va_index,
    flags,
    cfg,
    *,
    max_rounds=2000,
    adaboost_max_rounds=500,
    patience=50,
):
    model_cfg = {**cfg, "model_params": {**cfg.get("model_params", {}), factory_name: dict(params)}}
    factory = get_model(factory_name)

    t0 = perf_counter()
    model, es_meta = fit_with_train_tail_early_stopping(
        factory,
        X_tr,
        y_tr,
        flags,
        model_cfg,
        issue_d=issue_tr,
        tail_fraction=0.10,
        patience=patience,
        max_rounds=max_rounds,
        adaboost_max_rounds=adaboost_max_rounds,
    )
    fit_seconds = perf_counter() - t0

    p_val = model.predict_proba(X_va)[:, 1]
    pred_df = pd.DataFrame({
        "loan_id": va_index.to_numpy(dtype="int64"),
        "y_true": y_va.to_numpy(dtype="int8"),
        "y_prob": p_val.astype("float64"),
        "split": "val",
    })

    metrics = evaluate(
        y_va,
        p_val,
        "val",
        save_name,
        None,
        cfg["cost_ratios"],
        None,
        cfg.get("bootstrap_iterations", 250),
        cfg["seed"],
    )
    run_meta = {
        "family": family,
        "factory": factory_name,
        "params": params,
        **es_meta,
        "fit_seconds": round(fit_seconds, 2),
        "val_roc_auc": metrics["roc_auc"],
        "val_pr_auc": metrics["pr_auc"],
        "val_brier": metrics["brier"],
        "val_log_loss": metrics["log_loss"],
        "val_ece": metrics["ece"],
        "val_ks": metrics["ks"],
        "calibration_slope": metrics["calibration_slope"],
        "calibration_intercept": metrics["calibration_intercept"],
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    save_p3_artifacts(
        save_name,
        model,
        pred_df,
        "val",
        run_meta=run_meta,
        p3_root=P3_ROOT,
        brief_root=".",
        timestamp=ts,
    )
    write_json(P3_ROOT / "runs" / f"val_metrics_{save_name}.json", {**metrics, **run_meta})
    print(
        f"[T3] {save_name:20s} fit_rows={es_meta['fit_rows']:6d} "
        f"best_iter={es_meta['best_iteration']:4d} refit_rounds={es_meta['refit_rounds']:4d} "
        f"fit_s={fit_seconds:6.1f} val_roc={metrics['roc_auc']:.4f} val_pr={metrics['pr_auc']:.4f}",
        flush=True,
    )
    return run_meta


def run_family(family, cfg, trials=30, timeout_seconds=1500, fit_row_limit=25000):
    set_seed(cfg["seed"])
    tr, va, cols, flags, X_tr, y_tr, X_va, y_va = _load_train_val(cfg)

    cv_caps = {
        "lightgbm": (2000, 500),
        "xgboost": (2000, 500),
        "catboost": (1500, 500),
        "hgb": (600, 500),
        "gradient_boosting": (250, 500),
        "adaboost": (2000, 250),
    }
    max_cv, ada_cv = cv_caps[family]

    with threadpool_limits(limits=2):
        print(f"[T3] Starting Optuna tuning for {family} (trials={trials}, timeout={timeout_seconds}s)...", flush=True)
        params, rec = tune_booster_p3(
            family,
            X_tr,
            y_tr,
            cfg,
            trials=trials,
            timeout_seconds=timeout_seconds,
            fit_row_limit=fit_row_limit,
            max_rounds=max_cv,
            adaboost_max_rounds=ada_cv,
            patience=50,
            p3_root=P3_ROOT,
        )
        print(
            f"[T3] Tuned {family}: status={rec['status']} trials={rec['completed_trials']} "
            f"mean_cv_auc={rec['best_mean_auc']} elapsed={rec['elapsed_seconds']}s",
            flush=True,
        )

        if family == "lightgbm":
            for save_name, factory_name in [
                ("lgbm_native", "lgbm_native"),
                ("lgbm_native_flags", "lgbm_native_flags"),
                ("lgbm_imputed_flags", "lgbm_imputed_flags"),
            ]:
                _fit_and_score_val(
                    save_name,
                    factory_name,
                    "lightgbm",
                    params,
                    X_tr,
                    y_tr,
                    tr.issue_d,
                    X_va,
                    y_va,
                    va.index,
                    flags,
                    cfg,
                    max_rounds=2000,
                )
        elif family == "xgboost":
            _fit_and_score_val(
                "xgb_hist",
                "xgb_hist",
                "xgboost",
                params,
                X_tr,
                y_tr,
                tr.issue_d,
                X_va,
                y_va,
                va.index,
                flags,
                cfg,
                max_rounds=2000,
            )
        elif family == "hgb":
            _fit_and_score_val(
                "hgb",
                "hgb",
                "hgb",
                params,
                X_tr,
                y_tr,
                tr.issue_d,
                X_va,
                y_va,
                va.index,
                flags,
                cfg,
                max_rounds=1000,
            )
        elif family == "catboost":
            _fit_and_score_val(
                "catboost",
                "catboost_native",
                "catboost",
                params,
                X_tr,
                y_tr,
                tr.issue_d,
                X_va,
                y_va,
                va.index,
                flags,
                cfg,
                max_rounds=1500,
            )
        elif family == "gradient_boosting":
            sub_ids, sub_n = stratified_train_subsample(
                X_tr,
                y_tr,
                issue_d=tr.issue_d,
                n_rows=50000,
                seed=cfg["seed"],
                out_path=P3_ROOT / "tables" / "gb_subsample_ids.parquet",
            )
            _fit_and_score_val(
                "gb_classic",
                "gb_classic_50k",
                "gradient_boosting",
                params,
                X_tr.loc[sub_ids],
                y_tr.loc[sub_ids],
                tr.loc[sub_ids, "issue_d"],
                X_va,
                y_va,
                va.index,
                flags,
                cfg,
                max_rounds=300,
            )
        elif family == "adaboost":
            _fit_and_score_val(
                "adaboost",
                "adaboost_p3",
                "adaboost",
                params,
                X_tr,
                y_tr,
                tr.issue_d,
                X_va,
                y_va,
                va.index,
                flags,
                cfg,
                adaboost_max_rounds=350,
            )


def aggregate_tables():
    rec_df = rebuild_recovery_table(P3_ROOT)
    runs_dir = P3_ROOT / "runs"
    rows = []
    for p in sorted(runs_dir.glob("val_metrics_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        rows.append({
            "model": d["model"],
            "family": d["family"],
            "factory": d["factory"],
            "fit_rows": d["fit_rows"],
            "roc_auc": d["val_roc_auc"],
            "pr_auc": d["val_pr_auc"],
            "brier": d["val_brier"],
            "log_loss": d["val_log_loss"],
            "ece": d["val_ece"],
            "ks": d["val_ks"],
            "calibration_slope": d["calibration_slope"],
            "calibration_intercept": d["calibration_intercept"],
            "best_iteration": d["best_iteration"],
            "refit_rounds": d["refit_rounds"],
            "fit_seconds": d["fit_seconds"],
        })
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values("pr_auc", ascending=False).reset_index(drop=True)
    summary.to_csv(P3_ROOT / "tables" / "t3_val_summary.csv", index=False)

    lgbm_names = ["lgbm_native", "lgbm_native_flags", "lgbm_imputed_flags"]
    lgbm_df = summary.loc[summary["model"].isin(lgbm_names)].copy() if not summary.empty else pd.DataFrame()
    if not lgbm_df.empty:
        lgbm_df["arm_description"] = lgbm_df["model"].map({
            "lgbm_native": "Native NaN, no missing flags",
            "lgbm_native_flags": "Native NaN + 24 missing flags",
            "lgbm_imputed_flags": "Median imputed + 24 missing flags",
        })
        cols = [
            "model", "arm_description", "roc_auc", "pr_auc", "brier",
            "log_loss", "ece", "best_iteration", "refit_rounds", "fit_seconds",
        ]
        lgbm_df = lgbm_df[cols].sort_values("pr_auc", ascending=False).reset_index(drop=True)
    lgbm_df.to_csv(P3_ROOT / "tables" / "lgbm_arms_comparison.csv", index=False)
    print(f"[T3] Aggregated {len(rec_df)} recovery rows, {len(summary)} val summaries, {len(lgbm_df)} LGBM arms.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument(
        "--family",
        choices=["lightgbm", "xgboost", "hgb", "catboost", "gradient_boosting", "adaboost"],
    )
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=1500)
    parser.add_argument("--fit-row-limit", type=int, default=25000)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()

    cfg = config(args.config)
    if args.family:
        run_family(args.family, cfg, trials=args.trials, timeout_seconds=args.timeout, fit_row_limit=args.fit_row_limit)
    if args.aggregate:
        aggregate_tables()
