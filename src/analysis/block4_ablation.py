"""Analysis A - feature-group ablation (Block 4).

Leave-one-group-out refits of `xgb_hist` with P3's tuned hyper-parameters and P3's exact fitting recipe
(early stopping on the latest 10% of Train, refit on 100% of Train with round(best_iter / 0.9) rounds).
Every arm is fitted on Train and scored on Validation only.

Raw columns are dropped *before* UnderwritingFeatures, which only builds a ratio when both of its inputs
are present, so dependent ratios (e.g. loan_to_income when annual_inc is dropped) disappear with them.
Missing-indicator flags are restricted to the surviving columns.

The `full` arm must reproduce P3's frozen xgb_hist Validation predictions; it is the control for every delta.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.block3 import _xgb_estimator, fit_with_train_tail_early_stopping
from src.data.splits import load
from src.evaluation.stats_tests import Ranked, compare_models
from src.features.block2 import FEATURE_GROUPS, RATIOS, build_preprocessor
from src.models.baselines import model_frame

P3, P4 = Path("artifacts/p3"), Path("artifacts/p4")
OUT = P4 / "ablation"
RUN = sorted((Path("reports/runs")).glob("*_xgb_hist.json"))


def arms(cols):
    """name -> (raw columns to drop, build engineered ratios?)."""
    keep = set(cols)
    groups = {g: sorted(set(f) & keep) for g, f in FEATURE_GROUPS.items() if g != "engineered"}
    spec = {"full": ([], True)}
    for g, fields in groups.items():
        spec[f"drop_{g}"] = (fields, True)
    spec["drop_engineered"] = ([], False)
    # LC sets int_rate from sub_grade, so dropping lc_grade alone leaves the grade signal in int_rate.
    spec["drop_lc_grade_and_int_rate"] = (sorted(groups["lc_grade"] + ["int_rate"]), True)
    return spec


def xgb_params():
    if not RUN:
        raise FileNotFoundError("reports/runs/*_xgb_hist.json (P3's tuned parameters) not found")
    return json.loads(RUN[-1].read_text())["params"]


def factory_for(engineered, params):
    def factory(X, flags, cfg):
        prep = build_preprocessor("tree", flags=flags, impute_tree=False, engineered=engineered)
        return Pipeline([("preprocess", prep), ("model", _xgb_estimator(cfg["seed"], **params))])
    return factory


def run_arm(name, drop, engineered, data, flags_all, cfg, params):
    out = OUT / f"{name}.json"
    if out.exists():
        return json.loads(out.read_text())
    X_tr, y_tr, issue_d, X_va, y_va = data
    cols = [c for c in X_tr.columns if c not in set(drop)]
    flags = [f for f in flags_all if f in cols]
    start = time.perf_counter()
    pipe, meta = fit_with_train_tail_early_stopping(factory_for(engineered, params), X_tr[cols], y_tr, flags, cfg,
                                                    issue_d=issue_d, max_rounds=2000)
    p = pipe.predict_proba(X_va[cols])[:, 1]
    roc, pr, brier, ks = Ranked(y_va.to_numpy(), p).metrics(np.ones(len(p), dtype=int))
    ratios = [r for r, (a, b, _) in RATIOS.items() if engineered and a in cols and b in cols]
    result = dict(arm=name, dropped=list(drop), engineered_ratios=ratios, n_raw_inputs=len(cols),
                  n_model_inputs=int(len(pipe.named_steps["preprocess"].get_feature_names_out())),
                  n_flags=len(flags), val_roc_auc=float(roc), val_pr_auc=float(pr), val_brier=float(brier),
                  val_ks=float(ks), fit_seconds=round(time.perf_counter() - start, 1), **meta)
    pd.DataFrame({"loan_id": X_va.index, "y_true": y_va.to_numpy(), "y_prob": p}).to_parquet(
        OUT / f"{name}_val.parquet", index=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    return result


def run(cfg, only=None, n_bootstrap=1000):
    manifest = json.loads((P3 / "model_manifest.json").read_text())
    cols = manifest["feature_order"]
    frames = load(cfg, names=("train", "val"))
    tr, va = frames["train"], frames["val"]
    data = (model_frame(tr[cols]), tr["target"].astype(int), tr["issue_d"], model_frame(va[cols]),
            va["target"].astype(int))
    xgb = __import__("joblib").load(manifest["models"]["xgb_hist"]["path"])
    flags_all = list(xgb.named_steps["preprocess"].named_steps["columns"].transformers[-1][2])
    params = xgb_params()
    OUT.mkdir(parents=True, exist_ok=True)

    spec = arms(cols)
    results = {}
    for name, (drop, engineered) in spec.items():
        if only and name not in only and name != "full":
            continue
        print(f"[ablation] {name}: dropping {drop or ('engineered ratios' if not engineered else 'nothing')}",
              flush=True)
        results[name] = run_arm(name, drop, engineered, data, flags_all, cfg, params)
        print(f"[ablation] {name}: Val ROC {results[name]['val_roc_auc']:.4f} PR {results[name]['val_pr_auc']:.4f}"
              f" ({results[name]['fit_seconds']}s, {results[name]['refit_rounds']} rounds)", flush=True)

    # Control: the full refit must reproduce P3's frozen xgb_hist Validation predictions.
    full = pd.read_parquet(OUT / "full_val.parquet").set_index("loan_id")
    frozen = pd.read_parquet(manifest["models"]["xgb_hist"]["val_pred_path"]).set_index("loan_id").loc[full.index]
    reproduction_gap = float(np.abs(full.y_prob - frozen.y_prob).max())

    rows = []
    for name, res in results.items():
        row = {k: v for k, v in res.items() if k not in ("dropped",)}
        row["dropped"] = ";".join(res["dropped"])
        if name != "full":
            arm = pd.read_parquet(OUT / f"{name}_val.parquet").set_index("loan_id").loc[full.index]
            cmp = compare_models(full.y_true, arm.y_prob, full.y_prob, n_bootstrap=n_bootstrap, seed=cfg["seed"])
            for metric, key in (("roc_auc", "delong"), ("pr_auc", "paired_bootstrap"), ("brier", "paired_bootstrap")):
                src = cmp["delong"] if key == "delong" else cmp["paired_bootstrap"][metric]
                row[f"d_{metric}"] = src["difference"]
                row[f"d_{metric}_ci_low"], row[f"d_{metric}_ci_high"] = src["ci95"]
                row[f"d_{metric}_p"] = src["p_value"]
        rows.append(row)
    table = pd.DataFrame(rows)
    for metric in ("roc_auc", "pr_auc", "brier"):
        col = f"d_{metric}_p"
        if col in table:
            mask = table[col].notna()
            table.loc[mask, f"{col}_holm"] = holm(table.loc[mask, col].to_numpy())
    table["full_reproduces_p3_max_abs_diff"] = reproduction_gap
    table.to_csv(P4 / "tables" / "ablation.csv", index=False)
    plot(table)
    return table, reproduction_gap


def holm(p):
    """Holm step-down adjusted p-values (family = all ablation arms vs full)."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.minimum(1.0, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    out = np.empty_like(adj)
    out[order] = adj
    return out


def plot(table):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = table[table.arm != "full"]
    if t.empty or "d_roc_auc" not in t:
        return
    t = t.sort_values("d_roc_auc")
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.45 * len(t) + 1.8), sharey=True)
    for ax, metric, label in ((axes[0], "d_roc_auc", "Delta ROC-AUC vs full (DeLong 95% CI)"),
                              (axes[1], "d_pr_auc", "Delta PR-AUC vs full (paired bootstrap 95% CI)")):
        err = np.vstack([t[metric] - t[f"{metric}_ci_low"], t[f"{metric}_ci_high"] - t[metric]])
        ax.barh(t.arm, t[metric], xerr=err, color=np.where(t[metric] < 0, "#c0392b", "#2e86c1"), capsize=3)
        ax.axvline(0, c="grey", lw=0.8)
        ax.set_xlabel(label, fontsize=9)
    fig.suptitle("Analysis A - leave-one-group-out ablation, xgb_hist refit on Train, scored on Validation",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(P4 / "plots" / "ablation.png", dpi=140)
    plt.close(fig)
