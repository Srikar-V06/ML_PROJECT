"""Block 4 runner (P4): calibration, SHAP, temporal stability / sensitivity, feature-group ablation.

Development runs never open Test files. Test is only read by `--final`, which first writes
artifacts/p4/calibration_lock.json (calibrator digests, pre-registered method choice and thresholds) and
refuses to run twice. P3's frozen artifacts (artifacts/p3/**, test_lock.json) are read-only here.

    python -m src.block4 --stage calibration [--models catboost]
    python -m src.block4 --stage calibration --final
"""
import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.evaluation import calibration as cal  # noqa: E402
from src.evaluation.cost_sensitive import select  # noqa: E402
from src.evaluation.stats_tests import compare_models  # noqa: E402
from src.utils.io import config, digest, write_json  # noqa: E402

P3 = Path("artifacts/p3")
P4 = Path("artifacts/p4")
MANIFEST = P3 / "model_manifest.json"
LOCK = P4 / "calibration_lock.json"
CAL_MODELS = ["stack_lr", "catboost", "xgb_hist", "lgbm_native_flags", "adaboost", "catboost_imb_class_weights"]
SELECTION_RULE = "Lowest 5-fold cross-fitted Validation Brier among raw/platt/isotonic (ties -> simpler method)"


def manifest():
    return json.loads(MANIFEST.read_text())


def predictions(name, split):
    if split == "test" and not LOCK.exists():
        raise RuntimeError("Test predictions are only read by --final after the calibration lock is written")
    path = P3 / "predictions" / f"{name}_{split}.parquet"
    frame = pd.read_parquet(path).set_index("loan_id").sort_index()
    assert frame.index.is_unique and set(frame.y_true.unique()) == {0, 1}
    return frame


# ----------------------------------------------------------------------------- calibration (Step 10)

def reliability_plot(curves, title, path, bins=15):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, label="perfect")
    for label, (y, p) in curves.items():
        edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
        idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
        frame = pd.DataFrame(dict(y=y, p=p, b=idx)).groupby("b").mean()
        ax.plot(frame.p, frame.y, marker="o", ms=3, label=label)
    hi = max(float(np.quantile(p, .995)) for _, p in curves.values())
    ax.set(xlim=(0, min(1, hi * 1.1)), ylim=(0, min(1, hi * 1.1)), xlabel="Mean predicted default probability",
           ylabel="Observed default rate")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def calibration_dev(cfg, models):
    """Validation only: cross-fitted metrics, final calibrators fitted on all of Validation, method selection."""
    ratios, seed = cfg["cost_ratios"], cfg["seed"]
    frozen = manifest()["val_thresholds"]
    rows, selected = [], {}
    for name in models:
        val = predictions(name, "val")
        y, p = val.y_true.to_numpy(), val.y_prob.to_numpy()
        curves = {}
        for method in cal.METHODS:
            oof = cal.cross_fit(y, p, method, folds=5, seed=seed)
            # Tuned thresholds: raw uses P3's frozen manifest cutoffs; calibrated arms re-select on their OOF scores.
            tuned = frozen[name] if method == "raw" else select(y, oof, ratios)
            rows.append(dict(model=name, method=method, split="val_crossfit", **cal.summary(y, oof, ratios, tuned)))
            curves[method] = (y, oof)
            fitted = cal.make(method).fit(p, y)
            joblib.dump(fitted, P4 / "calibration" / f"{name}_{method}.joblib")
        reliability_plot(curves, f"{name}\nValidation, 5-fold cross-fitted",
                         P4 / "plots" / f"reliability_val_{name}.png")
        cand = [r for r in rows if r["model"] == name]
        best = min(cand, key=lambda r: (round(r["brier"], 6), cal.METHODS.index(r["method"])))
        selected[name] = best["method"]
    table = pd.DataFrame(rows)
    out = P4 / "tables" / "calibration_val.csv"
    if out.exists():  # keep rows for models not in this run
        old = pd.read_csv(out)
        table = pd.concat([old[~old.model.isin(models)], table], ignore_index=True)
    table.to_csv(out, index=False)
    sel_path = P4 / "calibration" / "selected_methods.json"
    prev = json.loads(sel_path.read_text()) if sel_path.exists() else {}
    write_json(sel_path, {**prev, **selected})
    return table, selected


def calibration_final(cfg):
    """One-shot Test read-out. Writes the lock first, then transforms the saved P3 Test probabilities."""
    if LOCK.exists():
        raise RuntimeError(f"{LOCK} exists: the Test read-out has already been performed")
    ratios, seed = cfg["cost_ratios"], cfg["seed"]
    selected = json.loads((P4 / "calibration" / "selected_methods.json").read_text())
    missing = set(CAL_MODELS) - set(selected)
    if missing:
        raise RuntimeError(f"Run the Validation stage for all models first; missing {sorted(missing)}")
    val_table = pd.read_csv(P4 / "tables" / "calibration_val.csv")
    frozen = manifest()["val_thresholds"]
    thresholds = {}
    for name in CAL_MODELS:
        thresholds[name] = {"raw_tuned": frozen[name], "bayes": cal.bayes_thresholds(ratios)}
        for method in ("platt", "isotonic"):
            r = val_table.query("model == @name and method == @method").iloc[0]
            thresholds[name][f"{method}_tuned"] = {str(float(k)): float(r[f"tuned_thr_{int(k)}"]) for k in ratios}
    calibrators = {f"{n}_{m}": digest(P4 / "calibration" / f"{n}_{m}.joblib") for n in CAL_MODELS for m in cal.METHODS}
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        sha = None
    write_json(LOCK, dict(
        locked_at=datetime.now(timezone.utc).isoformat(), git_sha=sha, selection_rule=SELECTION_RULE,
        selected_methods={n: selected[n] for n in CAL_MODELS}, thresholds=thresholds, calibrator_sha256=calibrators,
        p3_test_lock_sha256=digest(P3 / "test_lock.json"), p3_manifest_sha256=digest(MANIFEST),
        p3_test_predictions_sha256={n: digest(P3 / "predictions" / f"{n}_test.parquet") for n in CAL_MODELS}))

    rows, savings = [], []
    for name in CAL_MODELS:
        val, test = predictions(name, "val"), predictions(name, "test")
        y, raw = test.y_true.to_numpy(), test.y_prob.to_numpy()
        curves, scores = {}, {}
        for method in cal.METHODS:
            fitted = joblib.load(P4 / "calibration" / f"{name}_{method}.joblib")
            scores[method] = fitted.transform(raw)
            tuned = thresholds[name][f"{method}_tuned"] if method != "raw" else thresholds[name]["raw_tuned"]
            rows.append(dict(model=name, method=method, split="test", selected=method == selected[name],
                             **cal.summary(y, scores[method], ratios, tuned)))
            curves[method] = (y, scores[method])
        reliability_plot(curves, f"{name}\nTest (calibrators fitted on Validation)",P4 / "plots" / f"reliability_test_{name}.png")
        best = selected[name]
        cmp = compare_models(y, scores[best], raw, n_bootstrap=1000, seed=seed)["paired_bootstrap"]["brier"]
        res = {r["method"]: r for r in rows if r["model"] == name}
        for k in ratios:
            k = int(k)
            savings.append(dict(model=name, selected_method=best, cost_ratio=k,
                                raw_bayes_cost=res["raw"][f"bayes_cost_{k}"],
                                calibrated_bayes_cost=res[best][f"bayes_cost_{k}"],
                                p3_val_tuned_cost=res["raw"][f"tuned_cost_{k}"],
                                saving_vs_raw_bayes=res["raw"][f"bayes_cost_{k}"] - res[best][f"bayes_cost_{k}"],
                                gap_vs_p3_tuned=res[best][f"bayes_cost_{k}"] - res["raw"][f"tuned_cost_{k}"],
                                brier_diff_cal_minus_raw=cmp["difference"], brier_diff_ci95=cmp["ci95"],
                                brier_diff_p=cmp["p_value"]))
    pd.DataFrame(rows).to_csv(P4 / "tables" / "calibration_test.csv", index=False)
    pd.DataFrame(savings).to_csv(P4 / "tables" / "cost_savings.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--stage", choices=["calibration", "shap"], default="calibration")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--sample", type=int, default=5000, help="Validation rows for SHAP")
    parser.add_argument("--final", action="store_true", help="one-shot Test read-out (writes the lock)")
    args = parser.parse_args()
    cfg = config(args.config)
    for sub in ("tables", "plots", "calibration"):
        (P4 / sub).mkdir(parents=True, exist_ok=True)
    if args.stage == "calibration":
        if args.final:
            calibration_final(cfg)
        else:
            table, selected = calibration_dev(cfg, args.models or CAL_MODELS)
            cols = ["model", "method", "roc_auc", "brier", "log_loss", "ece", "cal_slope", "cal_intercept", "mean_p",
                    "bayes_cost_10", "tuned_cost_10", "bayes_thr_10", "tuned_thr_10"]
            print(table[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
            print("selected:", selected)
    elif args.stage == "shap":
        from src.analysis.block4_shap import run as run_shap
        imp, rho, checks, mono = run_shap(cfg, sample=args.sample)
        print(checks.to_string(index=False))
        print(imp.rank(ascending=False).sort_values("stack_lr").head(12).astype(int).to_string())
        print(rho.round(3).to_string())
        print(pd.DataFrame(mono).pivot(index="field", columns="model", values="spearman_value_vs_shap").round(3))


if __name__ == "__main__":
    main()
