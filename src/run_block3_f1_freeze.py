"""Step F1: Freeze the complete Block 3 candidate model set into artifacts/p3/model_manifest.json.

Strictly Validation-only: never loads or touches Test. Records file SHA-256 digests,
exact 50-feature order, fit_rows / subsample labels, Val-locked decision thresholds,
and locks the Champion model strictly by Validation PR AUC.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.splits import load
from src.evaluation.metrics import evaluate
from src.utils.io import artifact, config, digest, write_json

P3_ROOT = Path("artifacts/p3")


def _candidate_specs(cfg):
    art = Path(cfg["paths"]["artifacts"])
    p1_models = art / "models"
    p1_preds = art / "predictions"
    p2_models = art / "block2" / "models"
    p3_models = P3_ROOT / "models"
    p3_preds = P3_ROOT / "predictions"

    specs = [
        # P1 baselines (4)
        ("constant", "P1", "baseline", p1_models / "constant.joblib", p1_preds / "constant_val.parquet", 210000, "full_210k"),
        ("grade_only", "P1", "baseline", p1_models / "grade_only.joblib", p1_preds / "grade_only_val.parquet", 210000, "full_210k"),
        ("logreg_baseline", "P1", "linear", p1_models / "logreg_baseline.joblib", p1_preds / "logreg_baseline_val.parquet", 210000, "full_210k"),
        ("woe_scorecard", "P1", "scorecard", p1_models / "woe_scorecard.joblib", p1_preds / "woe_scorecard_val.parquet", 210000, "full_210k"),
        # P2 classical + best Val PCA variant per classifier (5)
        ("decision_tree", "P2", "tree", p2_models / "decision_tree.joblib", p1_preds / "decision_tree_val.parquet", 210000, "full_210k"),
        ("random_forest", "P2", "forest", p2_models / "random_forest.joblib", p1_preds / "random_forest_val.parquet", 210000, "full_210k"),
        ("pca_lr_95", "P2", "pca_linear", p2_models / "pca_lr_95.joblib", p1_preds / "pca_lr_95_val.parquet", 20000, "subsample_20k"),
        ("pca_random_forest_99", "P2", "pca_forest", p2_models / "pca_random_forest_99.joblib", p1_preds / "pca_random_forest_99_val.parquet", 20000, "subsample_20k"),
        ("pca_lightgbm_80", "P2", "pca_gbdt", p2_models / "pca_lightgbm_80.joblib", p1_preds / "pca_lightgbm_80_val.parquet", 20000, "subsample_20k"),
        # P3 boosters (8)
        ("xgb_hist", "P3", "xgboost", p3_models / "xgb_hist.joblib", p3_preds / "xgb_hist_val.parquet", 210000, "full_210k"),
        ("catboost", "P3", "catboost", p3_models / "catboost.joblib", p3_preds / "catboost_val.parquet", 210000, "full_210k"),
        ("lgbm_native_flags", "P3", "lightgbm", p3_models / "lgbm_native_flags.joblib", p3_preds / "lgbm_native_flags_val.parquet", 210000, "full_210k"),
        ("lgbm_native", "P3", "lightgbm", p3_models / "lgbm_native.joblib", p3_preds / "lgbm_native_val.parquet", 210000, "full_210k"),
        ("lgbm_imputed_flags", "P3", "lightgbm", p3_models / "lgbm_imputed_flags.joblib", p3_preds / "lgbm_imputed_flags_val.parquet", 210000, "full_210k"),
        ("hgb", "P3", "hgb", p3_models / "hgb.joblib", p3_preds / "hgb_val.parquet", 210000, "full_210k"),
        ("gb_classic", "P3", "gradient_boosting", p3_models / "gb_classic.joblib", p3_preds / "gb_classic_val.parquet", 50000, "subsample_50k"),
        ("adaboost", "P3", "adaboost", p3_models / "adaboost.joblib", p3_preds / "adaboost_val.parquet", 210000, "full_210k"),
        # P3 imbalance arms (6 non-duplicate; *_imb_none identical to xgb_hist / catboost)
        ("xgboost_imb_class_weights", "P3", "imbalance", p3_models / "xgboost_imb_class_weights.joblib", p3_preds / "xgboost_imb_class_weights_val.parquet", 210000, "weighted_210k"),
        ("xgboost_imb_undersample", "P3", "imbalance", p3_models / "xgboost_imb_undersample.joblib", p3_preds / "xgboost_imb_undersample_val.parquet", 74290, "undersample_74k"),
        ("xgboost_imb_smote", "P3", "imbalance", p3_models / "xgboost_imb_smote.joblib", p3_preds / "xgboost_imb_smote_val.parquet", 345710, "smote_346k"),
        ("catboost_imb_class_weights", "P3", "imbalance", p3_models / "catboost_imb_class_weights.joblib", p3_preds / "catboost_imb_class_weights_val.parquet", 210000, "weighted_210k"),
        ("catboost_imb_undersample", "P3", "imbalance", p3_models / "catboost_imb_undersample.joblib", p3_preds / "catboost_imb_undersample_val.parquet", 74290, "undersample_74k"),
        ("catboost_imb_smote", "P3", "imbalance", p3_models / "catboost_imb_smote.joblib", p3_preds / "catboost_imb_smote_val.parquet", 345710, "smote_346k"),
        # P3 monotone + ensembles (3)
        ("lgbm_monotone", "P3", "lightgbm_monotone", p3_models / "lgbm_monotone.joblib", p3_preds / "lgbm_monotone_val.parquet", 210000, "full_210k"),
        ("avg_top3", "P3", "ensemble", p3_models / "avg_top3.joblib", p3_preds / "avg_top3_val.parquet", 210000, "full_210k"),
        ("stack_lr", "P3", "ensemble", p3_models / "stack_lr.joblib", p3_preds / "stack_lr_val.parquet", 210000, "full_210k_oof_168k"),
    ]
    return specs


def run(cfg):
    va = load(cfg, names=("val",))["val"]
    cols = pd.read_csv(artifact(cfg, "feature_audit.csv")).query("decision == 'keep'")["feature_name"].tolist()
    val_ids = va.index.to_numpy(dtype="int64")
    val_y = va.target.to_numpy(dtype="int8")

    models_entry = {}
    val_thresholds = {}
    val_rows = []

    for name, block, family, model_path, pred_path, fit_rows, sub_label in _candidate_specs(cfg):
        if not model_path.exists() or not pred_path.exists():
            print(f"[F1] SKIPPING incomplete candidate: {name}", flush=True)
            continue
        pred = pd.read_parquet(pred_path)
        assert np.array_equal(pred["loan_id"].to_numpy(dtype="int64"), val_ids), f"ID mismatch for {name}"
        assert np.array_equal(pred["y_true"].to_numpy(dtype="int8"), val_y), f"y_true mismatch for {name}"
        p_val = pred["y_prob"].to_numpy(dtype="float64")

        m = evaluate(val_y, p_val, "val", name, None, cfg["cost_ratios"], None, 250, cfg["seed"])
        val_thresholds[name] = m["thresholds"]
        models_entry[name] = {
            "path": str(model_path),
            "sha256": digest(model_path),
            "bytes": int(model_path.stat().st_size),
            "block": block,
            "family": family,
            "fit_rows": int(fit_rows),
            "subsample_label": sub_label,
            "val_pred_path": str(pred_path),
            "val_roc_auc": float(m["roc_auc"]),
            "val_pr_auc": float(m["pr_auc"]),
            "val_brier": float(m["brier"]),
            "val_ece": float(m["ece"]),
            "val_threshold_10": float(m["thresholds"]["10.0"]),
        }
        val_rows.append({
            "model": name,
            "block": block,
            "family": family,
            "fit_rows": int(fit_rows),
            "subsample_label": sub_label,
            "val_roc_auc": float(m["roc_auc"]),
            "val_pr_auc": float(m["pr_auc"]),
            "val_brier": float(m["brier"]),
            "val_ece": float(m["ece"]),
            "val_t5": float(m["thresholds"]["5.0"]),
            "val_t10": float(m["thresholds"]["10.0"]),
            "val_t15": float(m["thresholds"]["15.0"]),
        })

    val_df = pd.DataFrame(val_rows).sort_values("val_pr_auc", ascending=False).reset_index(drop=True)
    champion = str(val_df.iloc[0]["model"])

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_order": cols,
        "champion": champion,
        "champion_selection_rule": "Highest Validation PR AUC (Average Precision) among all frozen candidates",
        "val_thresholds": val_thresholds,
        "models": models_entry,
    }
    manifest_path = P3_ROOT / "model_manifest.json"
    write_json(manifest_path, manifest)
    val_df.to_csv(P3_ROOT / "tables" / "frozen_val_leaderboard.csv", index=False)
    print(f"[F1] Froze {len(models_entry)} models in {manifest_path}. Champion (Val PR AUC): {champion} "
          f"(Val PR={val_df.iloc[0]['val_pr_auc']:.4f}, Val ROC={val_df.iloc[0]['val_roc_auc']:.4f})", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/base.yaml")
    run(config(parser.parse_args().config))
