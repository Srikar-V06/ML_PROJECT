"""Step T5: monotone-constraint LightGBM (lgbm_monotone).

1. Train-only decile check of each candidate sign; contradicted/weak signs are dropped.
2. Fit with chronological train-tail early stopping, same tuned params and inputs as lgbm_native.
3. Score Val only; compare against lgbm_native; verify monotonicity with an ICE sweep.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from src.block3 import DEFAULT_MONOTONE_SIGNS, fit_with_train_tail_early_stopping, save_p3_artifacts
from src.data.splits import load
from src.evaluation.metrics import evaluate
from src.features.block2 import UnderwritingFeatures
from src.models.baselines import model_frame
from src.models.registry import get_model
from src.utils.io import artifact, config, write_json
from src.utils.seed import set_seed

P3_ROOT = Path("artifacts/p3")
RHO_THRESHOLD = 0.5  # sign kept only if Spearman(decile rank, default rate) * expected >= 0.5


def sign_check(X_tr, y_tr):
    eng = UnderwritingFeatures().fit_transform(X_tr)
    summary, deciles = [], []
    for feat, expected in DEFAULT_MONOTONE_SIGNS.items():
        s = eng[feat]
        ok = s.notna()
        q = pd.qcut(s[ok], 10, labels=False, duplicates="drop")
        grp = pd.DataFrame({"x": s[ok], "y": y_tr[ok], "bin": q}).groupby("bin")
        table = grp.agg(x_min=("x", "min"), x_max=("x", "max"), rows=("y", "size"), default_rate=("y", "mean"))
        rho = float(spearmanr(table.index, table["default_rate"]).statistic)
        keep = bool(rho * expected >= RHO_THRESHOLD)
        summary.append({
            "feature": feat,
            "expected_sign": expected,
            "bins": int(len(table)),
            "first_bin_rate": float(table["default_rate"].iloc[0]),
            "last_bin_rate": float(table["default_rate"].iloc[-1]),
            "spearman_rho": rho,
            "missing_share": float((~ok).mean()),
            "missing_default_rate": float(y_tr[~ok].mean()) if (~ok).any() else None,
            "decision": "keep" if keep else "drop (0)",
            "applied_sign": expected if keep else 0,
        })
        deciles.append(table.reset_index().assign(feature=feat))
    return pd.DataFrame(summary), pd.concat(deciles, ignore_index=True)


def write_sign_md(summary, deciles, path):
    lines = [
        "# Monotone-constraint sign check (Train only)",
        "",
        "Each candidate feature is cut into Train deciles (`pd.qcut`, ties dropped; missing values excluded "
        "and reported separately). The sign is kept only if the Spearman correlation between decile rank and "
        f"decile default rate has the expected sign with magnitude >= {RHO_THRESHOLD}; otherwise the "
        "constraint is set to 0. Engineered ratios use P2's `UnderwritingFeatures` (zero/negative income -> missing). "
        "`fico_range_low` is the only FICO field in the 50-field set. All other features are unconstrained (0). "
        "Validation and Test were not used.",
        "",
        "| Feature | Expected | Bins | First-bin default | Last-bin default | Spearman rho | Missing share | Missing default | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in summary.itertuples():
        miss = "—" if r.missing_default_rate is None or pd.isna(r.missing_default_rate) else f"{r.missing_default_rate:.3f}"
        lines.append(
            f"| `{r.feature}` | {r.expected_sign:+d} | {r.bins} | {r.first_bin_rate:.3f} | {r.last_bin_rate:.3f} | "
            f"{r.spearman_rho:+.3f} | {r.missing_share:.2%} | {miss} | **{r.decision}** |"
        )
    lines += ["", "## Decile default rates", ""]
    for feat, t in deciles.groupby("feature", sort=False):
        rates = ", ".join(f"{v:.3f}" for v in t["default_rate"])
        lines.append(f"- `{feat}`: {rates}")
    lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def ice_sweep(model, X_tr, X_va, signs, out_png, seed, n_rows=2000, n_grid=25):
    prep = model.named_steps["preprocess"]
    est = model.named_steps["model"]
    names = list(prep.get_feature_names_out())
    rows = X_va.sample(n=min(n_rows, len(X_va)), random_state=seed)
    Xv = np.asarray(prep.transform(rows), dtype=float)
    Xt = np.asarray(prep.transform(X_tr), dtype=float)
    feats = [f for f, s in signs.items() if s != 0]
    results = []
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for ax, feat in zip(axes.flat, feats):
        j = names.index(f"numeric__{feat}")
        col = Xt[:, j]
        grid = np.unique(np.nanquantile(col[np.isfinite(col)], np.linspace(0.01, 0.99, n_grid)))
        curves = np.empty((len(Xv), len(grid)))
        for g, v in enumerate(grid):
            Xm = Xv.copy()
            Xm[:, j] = v
            curves[:, g] = est.predict_proba(Xm)[:, 1]
        diffs = np.diff(curves, axis=1) * signs[feat]
        violating_rows = int((diffs < -1e-10).any(axis=1).sum())
        results.append({
            "feature": feat, "level": "model_input", "sign": signs[feat], "grid_points": int(len(grid)),
            "rows": int(len(Xv)), "violating_rows": violating_rows,
            "max_violation": float(max(0.0, -diffs.min())),
        })
        for c in curves[:60]:
            ax.plot(grid, c, color="steelblue", alpha=0.15, linewidth=0.7)
        ax.plot(grid, curves.mean(axis=0), color="crimson", linewidth=2.2, label="mean (PD)")
        ax.set_title(f"{feat} (sign {signs[feat]:+d}) — violating rows: {violating_rows}/{len(Xv)}", fontsize=10)
        ax.set_xlabel(feat)
        ax.set_ylabel("P(default)")
        ax.grid(alpha=0.25)
    for ax in list(axes.flat)[len(feats):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("lgbm_monotone ICE sweep on 2,000 Val rows (one constrained model input varied, others fixed)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=140)
    plt.close(fig)

    # Raw-input check for annual_inc: changes loan_to_income, payment_to_income and the
    # UNconstrained revolving_to_income together, so raw-level monotonicity is not guaranteed.
    if signs.get("annual_inc", 0) != 0:
        raw_grid = np.unique(np.nanquantile(X_tr["annual_inc"].dropna(), np.linspace(0.01, 0.99, n_grid)))
        curves = np.empty((len(rows), len(raw_grid)))
        for g, v in enumerate(raw_grid):
            r = rows.copy()
            r["annual_inc"] = v
            curves[:, g] = model.predict_proba(r)[:, 1]
        diffs = np.diff(curves, axis=1) * signs["annual_inc"]
        results.append({
            "feature": "annual_inc", "level": "raw_input", "sign": signs["annual_inc"],
            "grid_points": int(len(raw_grid)), "rows": int(len(rows)),
            "violating_rows": int((diffs < -1e-10).any(axis=1).sum()),
            "max_violation": float(max(0.0, -diffs.min())),
        })
    return pd.DataFrame(results)


def run(cfg):
    set_seed(cfg["seed"])
    frames = load(cfg, names=("train", "val"))
    tr, va = frames["train"], frames["val"]
    cols = pd.read_csv(artifact(cfg, "feature_audit.csv")).query("decision == 'keep'")["feature_name"].tolist()
    X_tr, y_tr = model_frame(tr[cols]), tr.target
    X_va, y_va = model_frame(va[cols]), va.target

    summary, deciles = sign_check(X_tr, y_tr)
    (P3_ROOT / "tables").mkdir(parents=True, exist_ok=True)
    deciles.to_csv(P3_ROOT / "tables" / "monotone_sign_deciles.csv", index=False)
    signs = dict(zip(summary["feature"], summary["applied_sign"].astype(int)))
    print("[T5] Applied signs:", signs, flush=True)

    params = json.loads((P3_ROOT / "runs" / "tune_lightgbm.json").read_text())["params"]
    model_cfg = {**cfg, "model_params": {"lgbm_monotone": dict(params)}, "monotone_signs": signs}
    with threadpool_limits(limits=2):
        t0 = perf_counter()
        model, es_meta = fit_with_train_tail_early_stopping(
            get_model("lgbm_monotone"), X_tr, y_tr, (), model_cfg,
            issue_d=tr.issue_d, tail_fraction=0.10, patience=50, max_rounds=2000,
        )
        fit_seconds = perf_counter() - t0
        n_constrained = int(np.count_nonzero(model.named_steps["model"].get_params()["monotone_constraints"]))

        p_val = model.predict_proba(X_va)[:, 1]
        metrics = evaluate(y_va, p_val, "val", "lgbm_monotone", None, cfg["cost_ratios"], None,
                           cfg.get("bootstrap_iterations", 250), cfg["seed"])
        ice = ice_sweep(model, X_tr, X_va, signs, P3_ROOT / "plots" / "lgbm_monotone_ice.png", cfg["seed"])
    ice.to_csv(P3_ROOT / "tables" / "lgbm_monotone_ice_check.csv", index=False)

    pred_df = pd.DataFrame({"loan_id": va.index.to_numpy(dtype="int64"), "y_true": y_va.to_numpy(dtype="int8"),
                            "y_prob": p_val.astype("float64"), "split": "val"})
    run_meta = {
        "family": "lightgbm_monotone", "factory": "lgbm_monotone", "params": params, "monotone_signs": signs,
        "constrained_columns": n_constrained, **es_meta, "fit_seconds": round(fit_seconds, 2),
        "val_roc_auc": metrics["roc_auc"], "val_pr_auc": metrics["pr_auc"], "val_brier": metrics["brier"],
        "val_log_loss": metrics["log_loss"], "val_ece": metrics["ece"], "val_ks": metrics["ks"],
        "calibration_slope": metrics["calibration_slope"], "calibration_intercept": metrics["calibration_intercept"],
    }
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    save_p3_artifacts("lgbm_monotone", model, pred_df, "val", run_meta=run_meta, p3_root=P3_ROOT, brief_root=".", timestamp=ts)
    write_json(P3_ROOT / "runs" / "val_metrics_lgbm_monotone.json", {**metrics, **run_meta})

    base = json.loads((P3_ROOT / "runs" / "val_metrics_lgbm_native.json").read_text())
    comp = pd.DataFrame([
        {"model": "lgbm_native", "constraints": "none", "roc_auc": base["val_roc_auc"], "pr_auc": base["val_pr_auc"],
         "brier": base["val_brier"], "ece": base["val_ece"], "best_iteration": base["best_iteration"],
         "refit_rounds": base["refit_rounds"], "fit_seconds": base["fit_seconds"]},
        {"model": "lgbm_monotone", "constraints": json.dumps({k: v for k, v in signs.items() if v}),
         "roc_auc": metrics["roc_auc"], "pr_auc": metrics["pr_auc"], "brier": metrics["brier"], "ece": metrics["ece"],
         "best_iteration": es_meta["best_iteration"], "refit_rounds": es_meta["refit_rounds"],
         "fit_seconds": round(fit_seconds, 2)},
    ])
    comp["delta_roc_vs_native"] = comp["roc_auc"] - base["val_roc_auc"]
    comp["delta_pr_vs_native"] = comp["pr_auc"] - base["val_pr_auc"]
    comp.to_csv(P3_ROOT / "tables" / "monotone_comparison.csv", index=False)

    write_sign_md(summary, deciles, P3_ROOT / "tables" / "monotone_sign_check.md")
    with (P3_ROOT / "tables" / "monotone_sign_check.md").open("a", encoding="utf-8") as fh:
        fh.write("## Fit and verification (Val only)\n\n")
        fh.write(f"- Same tuned params and native-NaN/no-flag inputs as `lgbm_native`; "
                 f"`monotone_constraints_method='advanced'`; {n_constrained} constrained model columns.\n")
        fh.write(f"- Train-tail early stopping: best_iteration={es_meta['best_iteration']}, "
                 f"refit_rounds={es_meta['refit_rounds']} on {es_meta['fit_rows']:,} Train rows.\n")
        fh.write(f"- Val ROC AUC {metrics['roc_auc']:.4f} vs {base['val_roc_auc']:.4f} unconstrained "
                 f"(Δ {metrics['roc_auc'] - base['val_roc_auc']:+.4f}); "
                 f"PR AUC {metrics['pr_auc']:.4f} vs {base['val_pr_auc']:.4f} "
                 f"(Δ {metrics['pr_auc'] - base['val_pr_auc']:+.4f}).\n\n")
        fh.write("### ICE sweep (2,000 Val rows, 25-point Train-quantile grid)\n\n")
        fh.write("| Feature | Level | Sign | Grid | Violating rows | Max violation |\n|---|---|---:|---:|---:|---:|\n")
        for r in ice.itertuples():
            fh.write(f"| `{r.feature}` | {r.level} | {r.sign:+d} | {r.grid_points} | {r.violating_rows}/{r.rows} | {r.max_violation:.2e} |\n")
        fh.write("\n`model_input` varies one constrained column with all other model columns fixed; this is the "
                 "guarantee LightGBM enforces. `raw_input` varies raw `annual_inc`, which also moves "
                 "`loan_to_income`, `payment_to_income` and the unconstrained `revolving_to_income`, so raw-level "
                 "monotonicity in income is not guaranteed by construction.\n")
    print(f"[T5] lgbm_monotone val_roc={metrics['roc_auc']:.4f} val_pr={metrics['pr_auc']:.4f} "
          f"(native {base['val_roc_auc']:.4f}/{base['val_pr_auc']:.4f}) fit_s={fit_seconds:.1f}", flush=True)
    print(ice.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/base.yaml")
    run(config(parser.parse_args().config))
