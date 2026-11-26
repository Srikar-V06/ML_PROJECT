"""Step T4 runner: Imbalance study (none, class_weights, undersample, smote)
on the top-2 booster families ranked by Val Average Precision (PR AUC).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline as SkPipeline
from threadpoolctl import threadpool_limits

from src.block3 import (
    _fit_estimator_with_early_stop,
    _round_param_name,
    chronological_train_tail_split,
    extract_keep_flags,
    fit_with_train_tail_early_stopping,
    save_p3_artifacts,
)
from src.data.splits import load
from src.evaluation.metrics import evaluate
from src.features.block2 import build_preprocessor
from src.models.baselines import model_frame
from src.models.registry import get_model
from src.utils.io import artifact, config, write_json
from src.utils.seed import set_seed

P3_ROOT = Path("artifacts/p3")


def _select_top2_families():
    summary_path = P3_ROOT / "tables" / "t3_val_summary.csv"
    df = pd.read_csv(summary_path).sort_values("pr_auc", ascending=False)
    selected = []
    seen_families = set()
    for _, row in df.iterrows():
        fam = row["family"]
        if fam in seen_families:
            continue
        seen_families.add(fam)
        selected.append((fam, row["model"], row["factory"]))
        if len(selected) == 2:
            break
    if len(selected) < 2:
        raise RuntimeError(f"Expected at least 2 completed families in {summary_path}, found {len(selected)}")
    return selected


def _weight_params(family, y_tr):
    n_pos = int((y_tr == 1).sum())
    n_neg = int((y_tr == 0).sum())
    ratio = float(n_neg / max(1, n_pos))
    if family in {"lightgbm", "hgb"}:
        return {"class_weight": "balanced"}
    if family == "xgboost":
        return {"scale_pos_weight": ratio}
    if family == "catboost":
        return {"auto_class_weights": "Balanced"}
    if family == "adaboost":
        return {}
    return {}


def _build_resampling_estimator(family, factory_name, params, X_ref, flags, cfg, *, use_onehot_catboost=False):
    model_cfg = {**cfg, "model_params": {**cfg.get("model_params", {}), factory_name: dict(params)}}
    if use_onehot_catboost and family == "catboost":
        from catboost import CatBoostClassifier
        est = CatBoostClassifier(
            random_seed=cfg["seed"],
            thread_count=2,
            verbose=False,
            allow_writing_files=False,
            **params,
        )
        prep = build_preprocessor("tree", flags=flags, impute_tree=True)
        return prep, est
    pipe = get_model(factory_name)(X_ref, flags, model_cfg)
    return pipe.named_steps["preprocess"], pipe.named_steps["model"]


def _run_resampled_arm(
    arm,
    family,
    factory_name,
    params,
    X_tr,
    y_tr,
    issue_tr,
    flags,
    cfg,
    *,
    max_rounds=2000,
    patience=50,
):
    fit_idx, tail_idx = chronological_train_tail_split(X_tr, y_tr, issue_d=issue_tr, tail_fraction=0.10)
    X_fit, y_fit = X_tr.loc[fit_idx], y_tr.loc[fit_idx]
    X_tail, y_tail = X_tr.loc[tail_idx], y_tr.loc[tail_idx]

    use_onehot_cb = (arm == "smote" and family == "catboost")
    probe_prep, probe_est = _build_resampling_estimator(
        family, factory_name, params, X_fit, flags, cfg, use_onehot_catboost=use_onehot_cb
    )
    if arm == "smote" and not use_onehot_cb:
        probe_prep = build_preprocessor("tree", flags=flags, impute_tree=True)

    X_fit_t = probe_prep.fit_transform(X_fit, y_fit)
    X_tail_t = probe_prep.transform(X_tail)

    if arm == "undersample":
        sampler = RandomUnderSampler(sampling_strategy=1.0, random_state=cfg["seed"])
    elif arm == "smote":
        sampler = SMOTE(random_state=cfg["seed"])
    else:
        raise ValueError(arm)

    X_res, y_res = sampler.fit_resample(X_fit_t, y_fit)
    best_iter = _fit_estimator_with_early_stop(
        probe_est,
        X_res,
        y_res,
        X_tail_t,
        y_tail,
        patience=patience,
        max_rounds=max_rounds,
    )
    scaled_rounds = max(1, int(round(best_iter / 0.9)))

    # Final refit on 100% Train inside imblearn Pipeline (sampler runs ONLY in fit)
    final_prep, final_est = _build_resampling_estimator(
        family, factory_name, params, X_tr, flags, cfg, use_onehot_catboost=use_onehot_cb
    )
    if arm == "smote" and not use_onehot_cb:
        final_prep = build_preprocessor("tree", flags=flags, impute_tree=True)

    round_key = _round_param_name(final_est)
    clear_es = {}
    if type(final_est).__name__ == "XGBClassifier":
        clear_es["early_stopping_rounds"] = None
    elif type(final_est).__name__ == "HistGradientBoostingClassifier":
        clear_es["early_stopping"] = False
    final_est.set_params(**{round_key: scaled_rounds, **clear_es})

    final_sampler = (
        RandomUnderSampler(sampling_strategy=1.0, random_state=cfg["seed"])
        if arm == "undersample"
        else SMOTE(random_state=cfg["seed"])
    )
    prep_steps = list(final_prep.steps) if isinstance(final_prep, SkPipeline) else [("preprocess", final_prep)]
    final_pipe = ImbPipeline([*prep_steps, ("sampler", final_sampler), ("model", final_est)])
    final_pipe.fit(X_tr, y_tr)

    # Compute exact full-Train resampled row count for reporting
    n_pos = int((y_tr == 1).sum())
    n_neg = int((y_tr == 0).sum())
    resampled_rows = 2 * n_pos if arm == "undersample" else 2 * n_neg

    return final_pipe, {
        "best_iteration": int(best_iter),
        "refit_rounds": int(scaled_rounds),
        "fit_rows": int(len(X_tr)),
        "resampled_rows": int(resampled_rows),
    }


def _plot_imbalance_study(df, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    families = list(dict.fromkeys(df["family"]))
    arms = ["none", "class_weights", "undersample", "smote"]
    arm_labels = ["None", "Class Weights", "Undersample", "SMOTE"]
    x = np.arange(len(arms))
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    panels = [
        ("pr_auc", "Validation PR AUC (AP)"),
        ("roc_auc", "Validation ROC AUC"),
        ("brier", "Validation Brier Score (lower is better)"),
        ("mean_pred_prob", "Mean Predicted Probability vs Val Base Rate"),
    ]
    colors = ["#1f77b4", "#ff7f0e"]

    for ax, (col, title) in zip(axes.flat, panels):
        for i, fam in enumerate(families):
            sub = df.loc[df["family"] == fam].set_index("arm").reindex(arms)
            vals = sub[col].to_numpy(dtype=float)
            offset = (i - 0.5 * (len(families) - 1)) * width
            bars = ax.bar(x + offset, vals, width=width, label=fam, color=colors[i % len(colors)], alpha=0.88)
            for bar, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        v,
                        f"{v:.4f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
        if col == "mean_pred_prob":
            base_rate = float(df["val_base_rate"].iloc[0])
            ax.axhline(base_rate, color="crimson", linestyle="--", linewidth=1.5, label=f"Val base rate ({base_rate:.4f})")
        ax.set_xticks(x)
        ax.set_xticklabels(arm_labels)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("Step T4: Class Imbalance Study on Top-2 Booster Families (Full 210k Train, Val Evaluation)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def run(cfg):
    set_seed(cfg["seed"])
    frames = load(cfg, names=("train", "val"))
    tr, va = frames["train"], frames["val"]
    audit = pd.read_csv(artifact(cfg, "feature_audit.csv"))
    cols = audit.loc[audit.decision.eq("keep"), "feature_name"].tolist()
    flags = extract_keep_flags(artifact(cfg, "missing_value_table.csv"), cols)
    X_tr = model_frame(tr[cols])
    y_tr = tr.target
    X_va = model_frame(va[cols])
    y_va = va.target
    val_base_rate = float(y_va.mean())

    top2 = _select_top2_families()
    print(f"[T4] Selected top-2 families by Val PR AUC: {top2}", flush=True)

    rows = []
    with threadpool_limits(limits=2):
        for family, base_model, factory_name in top2:
            tune_path = P3_ROOT / "runs" / f"tune_{family}.json"
            base_params = json.loads(tune_path.read_text(encoding="utf-8"))["params"]
            max_rounds = 1500 if family == "catboost" else (1000 if family == "hgb" else 2000)

            for arm in ("none", "class_weights", "undersample", "smote"):
                save_name = f"{family}_imb_{arm}"
                notes = ""
                t0 = perf_counter()

                if arm == "none":
                    arm_flags = () if base_model == "lgbm_native" else flags
                    model_cfg = {**cfg, "model_params": {**cfg.get("model_params", {}), factory_name: dict(base_params)}}
                    model, es_meta = fit_with_train_tail_early_stopping(
                        get_model(factory_name),
                        X_tr,
                        y_tr,
                        arm_flags,
                        model_cfg,
                        issue_d=tr.issue_d,
                        tail_fraction=0.10,
                        patience=50,
                        max_rounds=max_rounds,
                    )
                    es_meta["resampled_rows"] = int(len(X_tr))
                    notes = f"Matches T3 {base_model}"
                elif arm == "class_weights":
                    arm_flags = () if base_model == "lgbm_native" else flags
                    w_params = {**base_params, **_weight_params(family, y_tr)}
                    model_cfg = {**cfg, "model_params": {**cfg.get("model_params", {}), factory_name: w_params}}
                    model, es_meta = fit_with_train_tail_early_stopping(
                        get_model(factory_name),
                        X_tr,
                        y_tr,
                        arm_flags,
                        model_cfg,
                        issue_d=tr.issue_d,
                        tail_fraction=0.10,
                        patience=50,
                        max_rounds=max_rounds,
                    )
                    es_meta["resampled_rows"] = int(len(X_tr))
                    notes = f"Weighted: {_weight_params(family, y_tr)}"
                else:
                    arm_flags = () if base_model == "lgbm_native" else flags
                    model, es_meta = _run_resampled_arm(
                        arm,
                        family,
                        factory_name,
                        base_params,
                        X_tr,
                        y_tr,
                        tr.issue_d,
                        arm_flags,
                        cfg,
                        max_rounds=max_rounds,
                        patience=50,
                    )
                    if arm == "smote" and family == "catboost":
                        notes = "One-hot + median imputation before plain SMOTE (CatBoost native categoricals incompatible with plain SMOTE)"
                    elif arm == "smote":
                        notes = "Median imputation + one-hot before plain SMOTE inside imblearn Pipeline"
                    else:
                        notes = "RandomUnderSampler(1.0) on Train fit rows only"

                fit_seconds = perf_counter() - t0
                p_val = model.predict_proba(X_va)[:, 1]
                pred_df = pd.DataFrame({
                    "loan_id": va.index.to_numpy(dtype="int64"),
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
                mean_prob = float(np.mean(p_val))
                row = {
                    "family": family,
                    "base_model": base_model,
                    "arm": arm,
                    "fit_rows": es_meta["fit_rows"],
                    "resampled_rows": es_meta["resampled_rows"],
                    "pr_auc": metrics["pr_auc"],
                    "roc_auc": metrics["roc_auc"],
                    "brier": metrics["brier"],
                    "log_loss": metrics["log_loss"],
                    "ece": metrics["ece"],
                    "calibration_slope": metrics["calibration_slope"],
                    "calibration_intercept": metrics["calibration_intercept"],
                    "mean_pred_prob": mean_prob,
                    "val_base_rate": val_base_rate,
                    "prob_ratio": mean_prob / val_base_rate,
                    "best_iteration": es_meta["best_iteration"],
                    "refit_rounds": es_meta["refit_rounds"],
                    "fit_seconds": round(fit_seconds, 2),
                    "notes": notes,
                }
                rows.append(row)

                ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                save_p3_artifacts(
                    save_name,
                    model,
                    pred_df,
                    "val",
                    run_meta=row,
                    p3_root=P3_ROOT,
                    brief_root=".",
                    timestamp=ts,
                )
                print(
                    f"[T4] {save_name:28s} pr={row['pr_auc']:.4f} roc={row['roc_auc']:.4f} "
                    f"brier={row['brier']:.4f} ece={row['ece']:.4f} mean_p={mean_prob:.4f} ({fit_seconds:.1f}s)",
                    flush=True,
                )

    out_df = pd.DataFrame(rows)
    out_csv = P3_ROOT / "tables" / "imbalance_study.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    _plot_imbalance_study(out_df, P3_ROOT / "plots" / "imbalance_study.png")
    print(f"[T4] Saved {out_csv} and imbalance_study.png", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/base.yaml")
    args = parser.parse_args()
    run(config(args.config))
