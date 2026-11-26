"""Step T6: ensembles.

--avg        Simple probability average of 3 diverse Val-selected members (avg_top3).
--oof NAME   Forward-chained out-of-fold scores for one stack base over the 4 expanding Train folds.
--meta       Logistic meta-learner on the OOF logits (168k out-of-time Train rows) -> stack_lr.

Members (pre-registered from Val PR AUC and diversity): xgb_hist (best GBDT), catboost
(different tree algorithm, native categoricals), logreg_baseline (P1 linear). Val and Test are
never used to fit anything here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from src.block3 import fit_with_train_tail_early_stopping, fold_missing_flags, save_p3_artifacts
from src.block3_ensembles import AveragingEnsemble, StackedEnsemble, to_logit
from src.data.splits import load
from src.evaluation.metrics import evaluate
from src.models.baselines import model_frame
from src.models.registry import get_model
from src.utils.io import artifact, config, write_json
from src.utils.seed import set_seed

P3_ROOT = Path("artifacts/p3")
STACK_DIR = P3_ROOT / "stack"

# name -> (fitted full-Train model path, Val prediction path, factory, tuned-params family or None, max_rounds)
MEMBERS = {
    "xgb_hist": ("artifacts/p3/models/xgb_hist.joblib", "artifacts/p3/predictions/xgb_hist_val.parquet",
                 "xgb_hist", "xgboost", 2000),
    "catboost": ("artifacts/p3/models/catboost.joblib", "artifacts/p3/predictions/catboost_val.parquet",
                 "catboost_native", "catboost", 1500),
    "logreg_baseline": ("artifacts/p1p2_300k/models/logreg_baseline.joblib",
                        "artifacts/p1p2_300k/predictions/logreg_baseline_val.parquet",
                        "logreg_baseline", None, None),
}


def _data(cfg):
    frames = load(cfg, names=("train", "val"))
    tr, va = frames["train"], frames["val"]
    cols = pd.read_csv(artifact(cfg, "feature_audit.csv")).query("decision == 'keep'")["feature_name"].tolist()
    return tr, va, cols, model_frame(tr[cols]), tr.target, model_frame(va[cols]), va.target


def _load_members(X_va):
    members, val_probs = [], {}
    for name, (model_path, pred_path, *_rest) in MEMBERS.items():
        model = joblib.load(model_path)
        p = model.predict_proba(X_va)[:, 1]
        ref = pd.read_parquet(pred_path)
        assert ref["loan_id"].to_numpy().tolist() == X_va.index.to_numpy().tolist(), name
        assert np.allclose(p, ref["y_prob"].to_numpy(), atol=1e-6), f"{name} does not reproduce saved Val predictions"
        members.append((name, model))
        val_probs[name] = p
    return members, val_probs


def _save(name, model, va, y_va, p_val, cfg, extra):
    metrics = evaluate(y_va, p_val, "val", name, None, cfg["cost_ratios"], None,
                       cfg.get("bootstrap_iterations", 250), cfg["seed"])
    pred_df = pd.DataFrame({"loan_id": va.index.to_numpy(dtype="int64"), "y_true": y_va.to_numpy(dtype="int8"),
                            "y_prob": np.asarray(p_val, dtype="float64"), "split": "val"})
    run_meta = {
        "family": "ensemble", "factory": name, "fit_rows": int(extra.pop("fit_rows")),
        "best_iteration": None, "refit_rounds": None, "fit_seconds": extra.pop("fit_seconds"),
        "val_roc_auc": metrics["roc_auc"], "val_pr_auc": metrics["pr_auc"], "val_brier": metrics["brier"],
        "val_log_loss": metrics["log_loss"], "val_ece": metrics["ece"], "val_ks": metrics["ks"],
        "calibration_slope": metrics["calibration_slope"], "calibration_intercept": metrics["calibration_intercept"],
        **extra,
    }
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    save_p3_artifacts(name, model, pred_df, "val", run_meta=run_meta, p3_root=P3_ROOT, brief_root=".", timestamp=ts)
    write_json(P3_ROOT / "runs" / f"val_metrics_{name}.json", {**metrics, **run_meta})
    print(f"[T6] {name:10s} val_roc={metrics['roc_auc']:.4f} val_pr={metrics['pr_auc']:.4f} "
          f"brier={metrics['brier']:.4f} ece={metrics['ece']:.4f}", flush=True)
    return metrics


def run_avg(cfg):
    set_seed(cfg["seed"])
    tr, va, cols, X_tr, y_tr, X_va, y_va = _data(cfg)
    t0 = perf_counter()
    members, val_probs = _load_members(X_va)
    ens = AveragingEnsemble(members)
    p = ens.predict_proba(X_va)[:, 1]
    assert np.allclose(p, np.column_stack(list(val_probs.values())).mean(axis=1))
    probs = pd.DataFrame(val_probs)
    config_rec = {
        "members": list(MEMBERS),
        "selection_rule": ("Best GBDT by Val PR AUC (xgb_hist) + CatBoost (distinct algorithm, native categoricals) "
                           "+ best P1/P2 linear/tree model by Val PR AUC (logreg_baseline). Unweighted mean of probabilities."),
        "member_val_pr_auc": {k: float(average_precision_score(y_va, v)) for k, v in val_probs.items()},
        "member_val_roc_auc": {k: float(roc_auc_score(y_va, v)) for k, v in val_probs.items()},
        "val_spearman_correlation": probs.corr(method="spearman").round(4).to_dict(),
        "note": "Members were chosen on Val, so the ensemble's Val score is mildly optimistic; Test stays untouched.",
    }
    write_json(P3_ROOT / "tables" / "avg_top3_config.json", config_rec)
    _save("avg_top3", ens, va, y_va, p, cfg,
          {"fit_rows": len(X_tr), "fit_seconds": round(perf_counter() - t0, 2), "members": list(MEMBERS)})


def run_oof(name, cfg):
    set_seed(cfg["seed"])
    tr, va, cols, X_tr, y_tr, X_va, y_va = _data(cfg)
    _, _, factory_name, family, max_rounds = MEMBERS[name]
    params = json.loads((P3_ROOT / "runs" / f"tune_{family}.json").read_text())["params"] if family else {}
    model_cfg = {**cfg, "model_params": {factory_name: dict(params)}}
    membership = pd.read_parquet(Path(cfg["paths"]["splits"]) / "train_expanding_cv.parquet")
    parts, log = [], []
    with threadpool_limits(limits=2):
        for fold, group in membership.groupby("fold", sort=True):
            fit_ids = group.loc[group.role.eq("train"), "id"].to_numpy()
            val_ids = group.loc[group.role.eq("val"), "id"].to_numpy()
            X_f, y_f = X_tr.loc[fit_ids], y_tr.loc[fit_ids]
            flags = fold_missing_flags(X_f)  # recomputed inside the fold's fitting rows only
            t0 = perf_counter()
            if family is None:
                model = get_model(factory_name)(X_f, flags, model_cfg).fit(X_f, y_f)
                meta = {"best_iteration": None, "refit_rounds": None}
            else:
                model, meta = fit_with_train_tail_early_stopping(
                    get_model(factory_name), X_f, y_f, flags, model_cfg,
                    issue_d=tr.issue_d.loc[fit_ids], tail_fraction=0.10, patience=50, max_rounds=max_rounds,
                )
            p = model.predict_proba(X_tr.loc[val_ids])[:, 1]
            y_v = y_tr.loc[val_ids]
            auc = float(roc_auc_score(y_v, p))
            parts.append(pd.DataFrame({"loan_id": val_ids.astype("int64"), "fold": int(fold),
                                       "y_true": y_v.to_numpy(dtype="int8"), "p": p}))
            log.append({"base": name, "fold": int(fold), "fit_rows": int(len(fit_ids)), "oof_rows": int(len(val_ids)),
                        "oof_roc_auc": auc, "best_iteration": meta["best_iteration"],
                        "refit_rounds": meta["refit_rounds"], "fit_seconds": round(perf_counter() - t0, 2)})
            print(f"[T6-OOF] {name} fold={fold} fit_rows={len(fit_ids)} oof_auc={auc:.4f} "
                  f"best_iter={meta['best_iteration']} ({log[-1]['fit_seconds']}s)", flush=True)
    STACK_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(parts, ignore_index=True).to_parquet(STACK_DIR / f"oof_{name}.parquet", index=False)
    write_json(STACK_DIR / f"oof_{name}.json", log)


def run_meta(cfg):
    set_seed(cfg["seed"])
    tr, va, cols, X_tr, y_tr, X_va, y_va = _data(cfg)
    oof = None
    for name in MEMBERS:
        part = pd.read_parquet(STACK_DIR / f"oof_{name}.parquet").rename(columns={"p": name})
        oof = part if oof is None else oof.merge(part, on=["loan_id", "fold", "y_true"], how="inner", validate="1:1")
    assert len(oof) == 168_000 and oof["loan_id"].is_unique, len(oof)
    assert oof["loan_id"].isin(tr.index).all() and not oof["loan_id"].isin(va.index).any()

    t0 = perf_counter()
    Z = to_logit(oof[list(MEMBERS)].to_numpy())
    meta = LogisticRegression(C=1.0, max_iter=1000, tol=1e-8).fit(Z, oof["y_true"])
    oof_stack = meta.predict_proba(Z)[:, 1]
    members, val_probs = _load_members(X_va)
    stack = StackedEnsemble(members, meta)
    p_val = stack.predict_proba(X_va)[:, 1]

    fold_rows = []
    for fold, g in oof.groupby("fold"):
        row = {"fold": int(fold), "rows": int(len(g))}
        for name in MEMBERS:
            row[f"{name}_oof_auc"] = float(roc_auc_score(g["y_true"], g[name]))
        fold_rows.append(row)
    weights = dict(zip(MEMBERS, map(float, meta.coef_[0])))
    meta_rec = {
        "meta_learner": "LogisticRegression(C=1.0) on member logits",
        "oof_rows": int(len(oof)), "oof_folds": sorted(map(int, oof["fold"].unique())),
        "coefficients": weights, "intercept": float(meta.intercept_[0]),
        "oof_stack_roc_auc_in_sample": float(roc_auc_score(oof["y_true"], oof_stack)),
        "oof_member_roc_auc": {n: float(roc_auc_score(oof["y_true"], oof[n])) for n in MEMBERS},
        "per_fold": fold_rows,
        "note": ("OOF scores come from members refit on each expanding fold's earlier loans and scored on the next "
                 "chunk; full-Train refits are the deployed members. Fold-1 members saw only 42k loans, so the meta "
                 "weights are learned on weaker members than those deployed."),
    }
    write_json(STACK_DIR / "stack_lr_meta.json", meta_rec)
    m = _save("stack_lr", stack, va, y_va, p_val, cfg,
              {"fit_rows": len(oof), "fit_seconds": round(perf_counter() - t0, 2), "members": list(MEMBERS),
               "meta_coefficients": weights, "meta_intercept": float(meta.intercept_[0])})

    rows = []
    for name, p in val_probs.items():
        rows.append({"model": name, "role": "member", "roc_auc": roc_auc_score(y_va, p),
                     "pr_auc": average_precision_score(y_va, p)})
    for name in ("avg_top3", "stack_lr"):
        d = json.loads((P3_ROOT / "runs" / f"val_metrics_{name}.json").read_text())
        rows.append({"model": name, "role": "ensemble", "roc_auc": d["roc_auc"], "pr_auc": d["pr_auc"],
                     "brier": d["brier"], "ece": d["ece"]})
    pd.DataFrame(rows).to_csv(P3_ROOT / "tables" / "ensembles_val.csv", index=False)
    print("[T6] meta coefficients:", weights, "intercept:", float(meta.intercept_[0]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--avg", action="store_true")
    parser.add_argument("--oof", choices=list(MEMBERS))
    parser.add_argument("--meta", action="store_true")
    args = parser.parse_args()
    cfg = config(args.config)
    if args.avg:
        run_avg(cfg)
    if args.oof:
        run_oof(args.oof, cfg)
    if args.meta:
        run_meta(cfg)
