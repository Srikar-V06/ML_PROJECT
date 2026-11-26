"""Step F3: Single locked Test-set benchmark evaluation.

CRITICAL GUARDRAIL:
- Calls verify_manifest_and_acquire_test_lock() before loading Test.
- Refuses execution if artifacts/p3/test_lock.json already exists.
- Loads Test ONCE via splits.load(cfg, names=("test",)).
- Uses Val-locked thresholds and 1,000-resample bootstrap CIs (overriding config's 250).
- Runs DeLong + paired bootstrap significance tests with Holm step-down adjustment.
- Writes benchmark.csv, significance.csv, Test predictions (canonical + brief), run JSON,
  and ROC/PR overlay + CI bar plots. Never touches artifacts/p1p2_300k/predictions/.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from src.block3 import (
    save_benchmark_table,
    to_brief_predictions,
    verify_manifest_and_acquire_test_lock,
)
from src.data.splits import load
from src.evaluation.metrics import evaluate
from src.evaluation.stats_tests import compare_models
from src.models.baselines import model_frame
from src.utils.io import artifact, config, write_json
from src.utils.seed import set_seed

P3_ROOT = Path("artifacts/p3")
BOOTSTRAP_FINAL = 1000

PREREGISTERED_PAIRS = [
    ("xgb_hist", "lgbm_native_flags"),
    ("catboost", "xgb_hist"),
    ("lgbm_native", "lgbm_monotone"),
    ("stack_lr", "catboost"),
    ("stack_lr", "avg_top3"),
    ("lgbm_native_flags", "logreg_baseline"),
    ("woe_scorecard", "logreg_baseline"),
]

OVERLAY_MODELS = [
    "stack_lr",
    "avg_top3",
    "catboost",
    "xgb_hist",
    "lgbm_native_flags",
    "lgbm_monotone",
    "hgb",
    "gb_classic",
    "adaboost",
    "random_forest",
    "logreg_baseline",
    "woe_scorecard",
    "grade_only",
    "decision_tree",
]


def holm_adjust(p_values):
    """Holm-Bonferroni step-down adjusted p-values (preserving input order)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return np.array([], dtype=float)
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    adj_ranked = np.empty(m, dtype=float)
    running_max = 0.0
    for i, val in enumerate(ranked):
        if np.isnan(val):
            adj_ranked[i] = np.nan
            continue
        cand = min(1.0, (m - i) * val)
        running_max = max(running_max, cand)
        adj_ranked[i] = running_max
    out = np.empty(m, dtype=float)
    out[order] = adj_ranked
    return out


def _threshold_metrics(y_true, y_prob, thr):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    pred_pos = p >= float(thr)  # matches interfaces.md: probability == threshold triggers rejection
    tp = int((pred_pos & (y == 1)).sum())
    fp = int((pred_pos & (y == 0)).sum())
    fn = int((~pred_pos & (y == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def _default_rate_at_approval(y_true, y_prob, approval_rate=0.80):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    n_app = max(1, int(round(approval_rate * len(y))))
    order = np.argsort(p, kind="mergesort")
    return float(y[order[:n_app]].mean())


def _plot_overlays(y_test, probs_map, bm_df, plots_dir):
    plots_dir.mkdir(parents=True, exist_ok=True)
    bm_idx = bm_df.set_index("model")
    chosen = [m for m in OVERLAY_MODELS if m in probs_map]
    cmap = plt.get_cmap("tab20")

    # 1. ROC overlay
    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    for i, name in enumerate(chosen):
        fpr, tpr, _ = roc_curve(y_test, probs_map[name])
        auc = bm_idx.loc[name, "roc_auc"]
        lw = 2.4 if name in {"stack_lr", "catboost", "xgb_hist"} else 1.4
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})", color=cmap(i % 20), linewidth=lw)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, alpha=0.6, label="Random (0.5000)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Test-Set ROC Curves (45,000 Held-Out Borrowers)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(plots_dir / "benchmark_roc_overlay.png", dpi=150)
    plt.close(fig)

    # 2. PR overlay
    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    for i, name in enumerate(chosen):
        prec, rec, _ = precision_recall_curve(y_test, probs_map[name])
        ap = bm_idx.loc[name, "pr_auc"]
        lw = 2.4 if name in {"stack_lr", "catboost", "xgb_hist"} else 1.4
        ax.plot(rec, prec, label=f"{name} (AP={ap:.4f})", color=cmap(i % 20), linewidth=lw)
    base = float(np.mean(y_test))
    ax.axhline(base, color="black", linestyle="--", linewidth=1.0, alpha=0.6, label=f"Base rate ({base:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Test-Set Precision-Recall Curves (45,000 Held-Out Borrowers)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(plots_dir / "benchmark_pr_overlay.png", dpi=150)
    plt.close(fig)

    # 3. CI bar charts (ROC AUC & PR AUC with 95% 1,000-resample bootstrap CIs)
    ordered = bm_df.sort_values("pr_auc", ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(ordered))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, max(7.5, 0.34 * len(ordered))), sharey=True)

    roc_err = np.vstack([ordered["roc_auc"] - ordered["roc_auc_ci_low"], ordered["roc_auc_ci_high"] - ordered["roc_auc"]])
    pr_err = np.vstack([ordered["pr_auc"] - ordered["pr_auc_ci_low"], ordered["pr_auc_ci_high"] - ordered["pr_auc"]])
    colors = ["#d62728" if b == "P3" else ("#2ca02c" if b == "P2" else "#1f77b4") for b in ordered["block"]]

    ax1.barh(y_pos, ordered["roc_auc"], xerr=roc_err, color=colors, alpha=0.85, ecolor="black", capsize=2.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(ordered["model"], fontsize=8.5)
    ax1.set_xlim(max(0.45, float(ordered["roc_auc_ci_low"].min()) - 0.02), min(1.0, float(ordered["roc_auc_ci_high"].max()) + 0.02))
    ax1.set_xlabel("Test ROC AUC (95% Bootstrap CI, B=1,000)")
    ax1.set_title("Test ROC AUC with 95% CIs")
    ax1.grid(axis="x", alpha=0.25)

    ax2.barh(y_pos, ordered["pr_auc"], xerr=pr_err, color=colors, alpha=0.85, ecolor="black", capsize=2.5)
    ax2.set_xlim(max(0.15, float(ordered["pr_auc_ci_low"].min()) - 0.02), min(0.60, float(ordered["pr_auc_ci_high"].max()) + 0.03))
    ax2.set_xlabel("Test PR AUC / AP (95% Bootstrap CI, B=1,000)")
    ax2.set_title("Test PR AUC with 95% CIs (Blue=P1, Green=P2, Red=P3)")
    ax2.grid(axis="x", alpha=0.25)

    fig.suptitle("Final Locked Test-Set Benchmark — All Frozen Models", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plots_dir / "benchmark_ci_bars.png", dpi=150)
    plt.close(fig)


def run(cfg, manifest_path="artifacts/p3/model_manifest.json", lock_path="artifacts/p3/test_lock.json"):
    set_seed(cfg["seed"])
    cols = pd.read_csv(artifact(cfg, "feature_audit.csv")).query("decision == 'keep'")["feature_name"].tolist()

    # 1. Verify SHA-256 of every model & exact feature order, then acquire one-shot Test lock.
    manifest = verify_manifest_and_acquire_test_lock(manifest_path, lock_path, expected_feature_order=cols)
    print(f"[F3] Acquired single-run Test lock at {lock_path} ({len(manifest['models'])} verified models).", flush=True)

    # 2. Load Test ONCE.
    te = load(cfg, names=("test",))["test"]
    X_test = model_frame(te[cols])
    y_test = te.target.to_numpy(dtype="int8")
    test_ids = te.index.to_numpy(dtype="int64")
    print(f"[F3] Loaded Test split ONCE: {len(y_test):,} borrowers (default rate {y_test.mean():.4f}).", flush=True)

    eff_path = P3_ROOT / "tables" / "efficiency.csv"
    eff_df = pd.read_csv(eff_path).set_index("model") if eff_path.exists() else pd.DataFrame()

    rows = []
    probs_map = {}
    p3_preds_dir = P3_ROOT / "predictions"
    brief_preds_dir = Path("reports/preds")
    p3_preds_dir.mkdir(parents=True, exist_ok=True)
    brief_preds_dir.mkdir(parents=True, exist_ok=True)

    for name, info in manifest["models"].items():
        model = joblib.load(info["path"])
        p_test = np.asarray(model.predict_proba(X_test)[:, 1], dtype="float64")
        probs_map[name] = p_test

        # Save Test predictions to artifacts/p3/predictions/ and reports/preds/ (NEVER P1/P2 predictions/)
        pred_df = pd.DataFrame({"loan_id": test_ids, "y_true": y_test, "y_prob": p_test, "split": "test"})
        pred_df.to_parquet(p3_preds_dir / f"{name}_test.parquet", index=False)
        to_brief_predictions(pred_df).to_parquet(brief_preds_dir / f"{name}_test.parquet", index=False)

        val_thrs = manifest["val_thresholds"][name]
        m = evaluate(
            y_test,
            p_test,
            "test",
            name,
            None,
            cfg["cost_ratios"],
            val_thresholds=val_thrs,
            n_bootstrap=BOOTSTRAP_FINAL,
            seed=cfg["seed"],
        )
        t10 = float(val_thrs["10.0"])
        prec10, rec10, f1_10 = _threshold_metrics(y_test, p_test, t10)
        dr80 = _default_rate_at_approval(y_test, p_test, 0.80)

        row = {
            "model": name,
            "block": info["block"],
            "family": info["family"],
            "fit_rows": info["fit_rows"],
            "subsample_label": info["subsample_label"],
            "roc_auc": m["roc_auc"],
            "roc_auc_ci_low": m["ci95"]["roc_auc"][0],
            "roc_auc_ci_high": m["ci95"]["roc_auc"][1],
            "pr_auc": m["pr_auc"],
            "pr_auc_ci_low": m["ci95"]["pr_auc"][0],
            "pr_auc_ci_high": m["ci95"]["pr_auc"][1],
            "ks": m["ks"],
            "brier": m["brier"],
            "brier_ci_low": m["ci95"]["brier"][0],
            "brier_ci_high": m["ci95"]["brier"][1],
            "log_loss": m["log_loss"],
            "ece": m["ece"],
            "calibration_slope": m["calibration_slope"],
            "calibration_intercept": m["calibration_intercept"],
            "val_threshold_10": t10,
            "precision_at_t10": prec10,
            "recall_at_t10": rec10,
            "f1_at_t10": f1_10,
            "approval_rate_at_t10": m["cost_per_1000"]["10.0"]["approval_rate"],
            "cost_per_1000_at_10": m["cost_per_1000"]["10.0"]["optimized"],
            "default_rate_at_80pct_approval": dr80,
        }
        if not eff_df.empty and name in eff_df.index:
            er = eff_df.loc[name]
            for col in (
                "train_seconds", "n_train_runs", "infer_ms_per_1000", "model_bytes", "model_mb",
                "train_tracemalloc_mb", "train_rss_mb", "n_trees", "total_leaves", "max_depth", "n_params",
            ):
                row[col] = er[col]
        rows.append(row)
        print(
            f"[F3] {name:28s} ROC={m['roc_auc']:.4f} [{m['ci95']['roc_auc'][0]:.4f},{m['ci95']['roc_auc'][1]:.4f}] "
            f"PR={m['pr_auc']:.4f} Brier={m['brier']:.4f} DR@80%={dr80:.4f}",
            flush=True,
        )

    bm_df = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    save_benchmark_table(bm_df, p3_root=P3_ROOT, brief_root=".")

    # 3. Significance tests: Champion-vs-all + pre-registered pairs, with Holm step-down adjustment.
    champion = manifest["champion"]
    sig_rows = []
    pair_groups = [
        ("champion_vs_all", [(champion, other) for other in manifest["models"] if other != champion]),
        ("preregistered", [(a, b) for a, b in PREREGISTERED_PAIRS if a in probs_map and b in probs_map]),
    ]
    for group_name, pairs in pair_groups:
        group_entries = []
        for a, b in pairs:
            res = compare_models(y_test, probs_map[a], probs_map[b], n_bootstrap=BOOTSTRAP_FINAL, seed=cfg["seed"])
            # DeLong ROC AUC
            dl = res["delong"]
            group_entries.append({
                "group": group_name, "model_a": a, "model_b": b, "test": "delong", "metric": "roc_auc",
                "difference": dl["difference"], "ci_low": dl["ci95"][0], "ci_high": dl["ci95"][1],
                "z": dl.get("z"), "p_value": dl["p_value"],
            })
            # Paired bootstrap (ROC AUC, PR AUC, Brier)
            for metric, vals in res["paired_bootstrap"].items():
                group_entries.append({
                    "group": group_name, "model_a": a, "model_b": b, "test": "paired_bootstrap", "metric": metric,
                    "difference": vals["difference"], "ci_low": vals["ci95"][0], "ci_high": vals["ci95"][1],
                    "z": None, "p_value": vals["p_value"],
                })
        gdf = pd.DataFrame(group_entries)
        gdf["p_holm"] = np.nan
        for (test_name, metric_name), idx in gdf.groupby(["test", "metric"]).groups.items():
            gdf.loc[idx, "p_holm"] = holm_adjust(gdf.loc[idx, "p_value"].to_numpy())
        sig_rows.append(gdf)

    sig_df = pd.concat(sig_rows, ignore_index=True)
    sig_p3 = P3_ROOT / "tables" / "significance.csv"
    sig_rep = Path("reports/tables/significance.csv")
    sig_rep.parent.mkdir(parents=True, exist_ok=True)
    sig_df.to_csv(sig_p3, index=False)
    sig_df.to_csv(sig_rep, index=False)

    # 4. Plots
    _plot_overlays(y_test, probs_map, bm_df, P3_ROOT / "plots")

    # 5. Run metadata JSON
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_payload = {
        "timestamp": ts,
        "champion": champion,
        "test_rows": int(len(y_test)),
        "test_default_rate": float(y_test.mean()),
        "bootstrap_iterations": BOOTSTRAP_FINAL,
        "models_evaluated": int(len(bm_df)),
        "top3_test_by_pr_auc": bm_df.head(3)[["model", "roc_auc", "pr_auc", "brier"]].to_dict(orient="records"),
    }
    write_json(P3_ROOT / "runs" / f"final_benchmark_{ts}.json", run_payload)
    write_json(Path("reports/runs") / f"final_benchmark_{ts}.json", run_payload)
    print(f"[F3] Final Test benchmark complete! Saved benchmark.csv, significance.csv, and overlay/CI plots.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--manifest", default="artifacts/p3/model_manifest.json")
    parser.add_argument("--lock", default="artifacts/p3/test_lock.json")
    args = parser.parse_args()
    run(config(args.config), args.manifest, args.lock)
