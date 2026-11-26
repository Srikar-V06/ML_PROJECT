"""Step F2: Analysis C — unified model efficiency profiling harness.

Measures every frozen model in model_manifest.json on the same machine in isolated
subprocesses:
- Training time: median of 3 refits if < 120 s, else n=1 (flagged).
- Inference latency per 1,000 rows: median of 3 runs pinned to 1 thread.
- Serialized joblib size on disk (bytes / MB).
- Peak Python memory (tracemalloc) and peak process RSS (macOS bytes -> MB).
- Structural complexity: n_trees, total_leaves, max_depth, n_params, subsample_label.
Strictly Train/Val only — NEVER loads Test.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter
import tracemalloc

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline as SkPipeline
from threadpoolctl import threadpool_limits

from src.block3 import extract_keep_flags
from src.block3_ensembles import AveragingEnsemble, StackedEnsemble
from src.data.splits import load
from src.models.baselines import model_frame
from src.utils.io import artifact, config

P3_ROOT = Path("artifacts/p3")
VAL_1K_CACHE = Path("/tmp/block3_f2_val_1k.parquet")

# Models whose single final refit is >= ~120s or slow -> n=1 explicitly flagged
SLOW_TRAIN_MODELS = {
    "catboost",
    "catboost_imb_class_weights",
    "catboost_imb_undersample",
    "catboost_imb_smote",
    "xgboost_imb_smote",
    "gb_classic",
    "adaboost",
    "stack_lr",
}


def _rss_mb():
    # On macOS (darwin), ru_maxrss is in bytes; on Linux it is in kilobytes.
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return ru / (1024.0 * 1024.0)
    return ru / 1024.0


def _set_threads_recursive(obj, n_threads):
    if isinstance(obj, (AveragingEnsemble, StackedEnsemble)):
        for _, m in obj.members:
            _set_threads_recursive(m, n_threads)
        return
    if hasattr(obj, "steps"):
        for _, step in obj.steps:
            _set_threads_recursive(step, n_threads)
        return
    cls = type(obj).__name__
    if cls == "CatBoostClassifier":
        try:
            obj.set_params(thread_count=n_threads)
        except Exception:
            pass
    elif hasattr(obj, "n_jobs"):
        try:
            obj.set_params(n_jobs=n_threads)
        except Exception:
            pass


def _extract_complexity(model):
    """Return (n_trees, total_leaves, max_depth, n_params) for any frozen model."""
    if isinstance(model, (AveragingEnsemble, StackedEnsemble)):
        trees = leaves = params = 0
        depth = 0
        for _, m in model.members:
            t, l, d, p = _extract_complexity(m)
            trees += t or 0
            leaves += l or 0
            depth = max(depth, d or 0)
            params += p or 0
        if isinstance(model, StackedEnsemble) and model.meta is not None:
            params += int(model.meta.coef_.size + model.meta.intercept_.size)
        return trees or None, leaves or None, depth or None, params or None

    # Unwrap pipeline and check for PCA intermediate
    pca_params = 0
    est = model
    if hasattr(model, "steps"):
        for _, step in model.steps:
            if type(step).__name__ == "PCA" and hasattr(step, "components_"):
                pca_params += int(step.components_.size)
        est = model.steps[-1][1]

    cls = type(est).__name__
    if cls == "DummyClassifier":
        return None, None, None, 1
    if cls == "LogisticRegression":
        return None, None, None, int(est.coef_.size + est.intercept_.size) + pca_params
    if cls == "Scorecard":
        # Count total scorecard bin weights + base
        n_bins = sum(len(getattr(b, "woe_", [])) for b in getattr(est, "binners", {}).values())
        return None, None, None, int(n_bins + 1)
    if cls == "DecisionTreeClassifier":
        return 1, int(est.get_n_leaves()), int(est.get_depth()), None
    if cls == "RandomForestClassifier":
        n_t = len(est.estimators_)
        leaves = sum(int(t.get_n_leaves()) for t in est.estimators_)
        depth = max(int(t.get_depth()) for t in est.estimators_)
        return n_t, leaves, depth, (pca_params or None)
    if cls == "AdaBoostClassifier":
        n_t = len(est.estimators_)
        leaves = sum(int(t.get_n_leaves()) for t in est.estimators_)
        depth = max(int(t.get_depth()) for t in est.estimators_)
        return n_t, leaves, depth, None
    if cls == "GradientBoostingClassifier":
        trees = est.estimators_.ravel()
        n_t = len(trees)
        leaves = sum(int(t.get_n_leaves()) for t in trees)
        depth = max(int(t.get_depth()) for t in trees)
        return n_t, leaves, depth, None
    if cls == "HistGradientBoostingClassifier":
        preds = [row[0] for row in est._predictors]
        n_t = len(preds)
        leaves = sum(int(p.get_n_leaf_nodes()) for p in preds)
        depth = max(int(p.get_max_depth()) for p in preds)
        return n_t, leaves, depth, None
    if cls == "LGBMClassifier":
        df = est.booster_.trees_to_dataframe()
        n_t = int(df["tree_index"].nunique())
        leaves = int(df["node_depth"].loc[df["left_child"].isna()].shape[0])
        depth = int(df["node_depth"].max())
        return n_t, leaves, depth, (pca_params or None)
    if cls == "XGBClassifier":
        df = est.get_booster().trees_to_dataframe()
        n_t = int(df["Tree"].nunique())
        leaves = int((df["Feature"] == "Leaf").sum())
        max_d = int(est.get_params().get("max_depth") or 6)
        return n_t, leaves, max_d, None
    if cls == "CatBoostClassifier":
        n_t = int(est.tree_count_)
        d = int(est.get_params().get("depth") or 6)
        leaves = n_t * (2 ** d)
        return n_t, leaves, d, None
    return None, None, None, (pca_params or None)


def _worker_infer(name, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    info = manifest["models"][name]
    X_1k = pd.read_parquet(VAL_1K_CACHE)
    model = joblib.load(info["path"])
    _set_threads_recursive(model, 1)

    with threadpool_limits(limits=1):
        # Warm-up (discarded)
        _ = model.predict_proba(X_1k)[:, 1]

        tracemalloc.start()
        times = []
        for _ in range(3):
            t0 = perf_counter()
            p = model.predict_proba(X_1k)[:, 1]
            times.append((perf_counter() - t0) * 1000.0 * (1000.0 / len(X_1k)))
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    n_trees, total_leaves, max_depth, n_params = _extract_complexity(model)
    out = {
        "infer_ms_per_1000": float(np.median(times)),
        "infer_runs_ms": [round(t, 3) for t in times],
        "infer_tracemalloc_mb": round(peak_bytes / (1024.0 * 1024.0), 3),
        "infer_rss_mb": round(_rss_mb(), 2),
        "n_trees": n_trees,
        "total_leaves": total_leaves,
        "max_depth": max_depth,
        "n_params": n_params,
    }
    print(json.dumps(out))


def _worker_train(name, manifest_path, cfg_path):
    cfg = config(cfg_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    info = manifest["models"][name]
    cols = manifest["feature_order"]
    model = joblib.load(info["path"])

    # Ensembles: training time = sum of base fits (or meta fit); avoid redundant multi-minute OOF refit
    if name == "avg_top3":
        # No training required (pure average of pre-fitted base models)
        out = {
            "train_seconds": 0.0,
            "train_runs_s": [0.0],
            "n_train_runs": 3,
            "train_tracemalloc_mb": 0.0,
            "train_rss_mb": round(_rss_mb(), 2),
            "train_timing_note": "No separate fit (unweighted average of 3 base models)",
        }
        print(json.dumps(out))
        return

    if name == "stack_lr":
        # Time the meta-learner fit on the 168k OOF logits (3 runs) + record OOF base wall time note
        from sklearn.linear_model import LogisticRegression
        from src.block3_ensembles import to_logit
        oof = None
        for mname in ("xgb_hist", "catboost", "logreg_baseline"):
            part = pd.read_parquet(P3_ROOT / "stack" / f"oof_{mname}.parquet").rename(columns={"p": mname})
            oof = part if oof is None else oof.merge(part, on=["loan_id", "fold", "y_true"], how="inner")
        Z = to_logit(oof[["xgb_hist", "catboost", "logreg_baseline"]].to_numpy())
        y_oof = oof["y_true"].to_numpy()
        tracemalloc.start()
        times = []
        with threadpool_limits(limits=4):
            for _ in range(3):
                t0 = perf_counter()
                LogisticRegression(C=1.0, max_iter=1000, tol=1e-8).fit(Z, y_oof)
                times.append(perf_counter() - t0)
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        # Total OOF base fit time from saved logs
        oof_s = 0.0
        for mname in ("xgb_hist", "catboost", "logreg_baseline"):
            logs = json.loads((P3_ROOT / "stack" / f"oof_{mname}.json").read_text())
            oof_s += sum(r["fit_seconds"] for r in logs)
        out = {
            "train_seconds": round(float(np.median(times)) + oof_s, 2),
            "train_runs_s": [round(t + oof_s, 2) for t in times],
            "n_train_runs": 1,
            "train_tracemalloc_mb": round(peak_bytes / (1024.0 * 1024.0), 3),
            "train_rss_mb": round(_rss_mb(), 2),
            "train_timing_note": f"n=1: 4-fold OOF base fits ({oof_s:.1f}s) + median-of-3 meta fit ({np.median(times):.2f}s)",
        }
        print(json.dumps(out))
        return

    tr = load(cfg, names=("train",))["train"]
    X_tr = model_frame(tr[cols])
    y_tr = tr.target

    if info["subsample_label"] == "subsample_20k":
        X_tr = X_tr.sample(20000, random_state=cfg["seed"])
        y_tr = y_tr.loc[X_tr.index]
    elif info["subsample_label"] == "subsample_50k":
        sub_ids = pd.read_parquet(P3_ROOT / "tables" / "gb_subsample_ids.parquet")["loan_id"].to_numpy()
        X_tr = X_tr.loc[sub_ids]
        y_tr = y_tr.loc[sub_ids]

    n_runs = 1 if name in SLOW_TRAIN_MODELS else 3
    times = []
    tracemalloc.start()
    with threadpool_limits(limits=4):
        for i in range(n_runs):
            if name == "woe_scorecard":
                from src.models.registry import get_model
                fresh = get_model("woe_scorecard")(X_tr, [], cfg)
            else:
                fresh = clone(model)
            _set_threads_recursive(fresh, 4)
            t0 = perf_counter()
            fresh.fit(X_tr, y_tr)
            dt = perf_counter() - t0
            times.append(dt)
            if i == 0 and dt >= 120.0:
                n_runs = 1
                break
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    note = f"median of {len(times)} refits (4 threads)" if len(times) == 3 else "n=1 refit (>=120s or slow family, 4 threads)"
    out = {
        "train_seconds": round(float(np.median(times)), 3),
        "train_runs_s": [round(t, 3) for t in times],
        "n_train_runs": len(times),
        "train_tracemalloc_mb": round(peak_bytes / (1024.0 * 1024.0), 3),
        "train_rss_mb": round(_rss_mb(), 2),
        "train_timing_note": note,
    }
    print(json.dumps(out))


def run_all(cfg_path):
    cfg = config(cfg_path)
    manifest_path = P3_ROOT / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cols = manifest["feature_order"]

    # Cache fixed seeded 1,000-row Val slice for inference workers
    va = load(cfg, names=("val",))["val"]
    X_val_1k = model_frame(va[cols]).sample(n=1000, random_state=cfg["seed"])
    X_val_1k.to_parquet(VAL_1K_CACHE)

    py = sys.executable
    env_infer = {**os.environ, "MPLCONFIGDIR": "/tmp/mpl", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    env_train = {**os.environ, "MPLCONFIGDIR": "/tmp/mpl", "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}

    rows = []
    for name, info in manifest["models"].items():
        r_inf = subprocess.run(
            [py, "-m", "src.run_block3_f2_efficiency", "--worker-infer", name, "--manifest", str(manifest_path)],
            capture_output=True, text=True, check=True, env=env_infer,
        )
        inf_data = json.loads(r_inf.stdout.strip().splitlines()[-1])

        r_tr = subprocess.run(
            [py, "-m", "src.run_block3_f2_efficiency", "--worker-train", name, "--manifest", str(manifest_path), "--config", cfg_path],
            capture_output=True, text=True, check=True, env=env_train,
        )
        tr_data = json.loads(r_tr.stdout.strip().splitlines()[-1])

        row = {
            "model": name,
            "block": info["block"],
            "family": info["family"],
            "fit_rows": info["fit_rows"],
            "subsample_label": info["subsample_label"],
            "train_seconds": tr_data["train_seconds"],
            "n_train_runs": tr_data["n_train_runs"],
            "train_timing_note": tr_data["train_timing_note"],
            "infer_ms_per_1000": round(inf_data["infer_ms_per_1000"], 3),
            "model_bytes": info["bytes"],
            "model_mb": round(info["bytes"] / (1024.0 * 1024.0), 4),
            "train_tracemalloc_mb": tr_data["train_tracemalloc_mb"],
            "train_rss_mb": tr_data["train_rss_mb"],
            "infer_tracemalloc_mb": inf_data["infer_tracemalloc_mb"],
            "infer_rss_mb": inf_data["infer_rss_mb"],
            "n_trees": inf_data["n_trees"],
            "total_leaves": inf_data["total_leaves"],
            "max_depth": inf_data["max_depth"],
            "n_params": inf_data["n_params"],
        }
        rows.append(row)
        print(
            f"[F2] {name:28s} train={row['train_seconds']:7.2f}s (n={row['n_train_runs']}) "
            f"infer={row['infer_ms_per_1000']:7.2f}ms/1k size={row['model_mb']:.3f}MB "
            f"rss={row['train_rss_mb']:.0f}MB",
            flush=True,
        )

    out_df = pd.DataFrame(rows)
    out_p3 = P3_ROOT / "tables" / "efficiency.csv"
    out_p3.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_p3, index=False)
    out_rep = Path("reports/tables/efficiency.csv")
    out_rep.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_rep, index=False)
    print(f"[F2] Saved {out_p3} ({len(out_df)} models)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--manifest", default="artifacts/p3/model_manifest.json")
    parser.add_argument("--worker-infer")
    parser.add_argument("--worker-train")
    args = parser.parse_args()

    if args.worker_infer:
        _worker_infer(args.worker_infer, args.manifest)
    elif args.worker_train:
        _worker_train(args.worker_train, args.manifest, args.config)
    else:
        run_all(args.config)
