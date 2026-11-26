# Handoff to P4 (Block 4 — Calibration & Additional Analyses A / D / F)

**Status:** Block 3 frozen (`tag: block3-freeze`). All 26 candidate models (P1 baselines, P2 classical/PCA, and P3 boosters, imbalance arms, monotone LightGBM, and ensembles) have been frozen on Validation and evaluated **once** on the 45,000-borrower held-out Test cohort under a cryptographic SHA-256 manifest lock.

> **CRITICAL GUARDRAILS FOR P4:**
> 1. **Do NOT delete `artifacts/p3/test_lock.json` or re-run `src/final_benchmark.py`.** The single Test benchmark run is locked. For P4 post-hoc calibration (Step 10), fit Platt / isotonic calibrators strictly on **Validation** predictions (`*_val.parquet`), transform **Test** predictions (`*_test.parquet`) once to report pre- vs post-calibration Brier/ECE/cost savings, and write P4 outputs to `artifacts/p4/`.
> 2. **Do NOT touch `artifacts/p1p2_300k/predictions/`.** `tests/test_p1p2_handoff.py` asserts exactly 21 `*_val.parquet` and 0 `*_test.parquet` files in that folder.

---

## 1. Frozen Manifest, Champion & Key Artifacts

| Artifact | Path | Description |
|---|---|---|
| **Frozen Model Manifest** | `artifacts/p3/model_manifest.json` | 26 models with file paths, SHA-256 digests, `bytes`, `fit_rows`, `subsample_label`, exact 50-column `feature_order`, Val metrics, and Val-locked `5:1 / 10:1 / 15:1` cost thresholds |
| **Single-Run Test Lock** | `artifacts/p3/test_lock.json` | Timestamp, git SHA, and manifest SHA-256 from the single Test benchmark run |
| **Final Test Benchmark** | `artifacts/p3/tables/benchmark.csv` (mirror: `reports/tables/benchmark.csv`) | All 26 models on Test (`45,000` rows): ROC-AUC, PR-AUC, KS, Brier, log-loss, ECE, cal slope/intercept, 1,000-resample 95% CIs, precision/recall/F1/cost at Val-locked `10:1` cutoff, DR@80% approval, timing, memory & complexity |
| **Statistical Significance** | `artifacts/p3/tables/significance.csv` (mirror: `reports/tables/significance.csv`) | DeLong + 1,000-resample paired bootstrap with **Holm step-down** `p_holm` for Champion-vs-all (25 pairs) and 7 pre-registered pairs |
| **Efficiency (Analysis C)** | `artifacts/p3/tables/efficiency.csv` (mirror: `reports/tables/efficiency.csv`) | Isolated subprocess timings (`train_seconds`, 1-thread `infer_ms_per_1000`), size, `tracemalloc` + RSS, tree/leaf/depth/parameter counts |
| **Imbalance Study** | `artifacts/p3/tables/imbalance_study.csv` · `artifacts/p3/plots/imbalance_study.png` | `xgboost` & `catboost` across `none`, `class_weights`, `undersample`, `smote` |
| **LightGBM & Monotone** | `artifacts/p3/tables/lgbm_arms_comparison.csv` · `monotone_sign_check.md` · `plots/lgbm_monotone_ice.png` | 3 missing-value arms + 7-feature monotone LightGBM audit & ICE sweep |
| **Block 3 Report & Slides** | `artifacts/p3/report_block3.md` · `docs/viva_qa.md` | Metric-backed Block 3 report sections, 3 slide outlines, and restored P1+P2+P3 Viva Q&A |

### Champion & Top Standalone Models

| Role | Model Name | Saved Pipeline (`.joblib`) | Val PR / ROC | Test PR / ROC | Test Brier / ECE | Val-Locked `10:1` Cutoff |
|---|---|---|---:|---:|---:|---:|
| **Champion (Overall)** | `stack_lr` | `artifacts/p3/models/stack_lr.joblib` | 0.4410 / 0.7529 | **0.4279** / 0.7439 | **0.1386** / **0.0045** | `0.101725` |
| **Best Standalone Booster** | `catboost` | `artifacts/p3/models/catboost.joblib` | 0.4400 / 0.7529 | 0.4276 / **0.7442** | 0.1395 / 0.0242 | `0.087400` |
| **Fast Pareto Booster** | `xgb_hist` | `artifacts/p3/models/xgb_hist.joblib` | 0.4401 / 0.7517 | 0.4264 / 0.7429 | 0.1402 / 0.0291 | `0.095776` |
| **Best LightGBM** | `lgbm_native_flags` | `artifacts/p3/models/lgbm_native_flags.joblib` | 0.4394 / 0.7512 | 0.4243 / 0.7425 | 0.1403 / 0.0279 | `0.086201` |
| **Monotone Booster** | `lgbm_monotone` | `artifacts/p3/models/lgbm_monotone.joblib` | 0.4366 / 0.7502 | 0.4235 / 0.7414 | 0.1407 / 0.0323 | `0.090176` |
| **Best P1 Baseline** | `logreg_baseline` | `artifacts/p1p2_300k/models/logreg_baseline.joblib` | 0.4307 / 0.7473 | 0.4157 / 0.7372 | 0.1406 / 0.0185 | `0.098881` |

---

## 2. Prediction Schemas & Model Loading for P4

Every Validation prediction file has the exact same `45,000` `loan_id`s (`splits/p1p2_300k/val_ids.parquet`), and every Test prediction file has the exact same `45,000` `loan_id`s (`splits/p1p2_300k/test_ids.parquet`).

- **Canonical schema** (`loan_id: int64, y_true: int8, y_prob: float64, split: str`):
  - **Val (P1/P2 models):** `artifacts/p1p2_300k/predictions/<name>_val.parquet`
  - **Val (P3 models):** `artifacts/p3/predictions/<name>_val.parquet` *(also listed in `model_manifest.json` under `models[name]["val_pred_path"]`)*
  - **Test (all 26 frozen models):** `artifacts/p3/predictions/<name>_test.parquet`
- **Brief mirror schema** (`id: int64, y_true: int8, p_default: float64`):
  - `reports/preds/<name>_{val,test}.parquet` (and `models/<name>.joblib` for P3 models).

### Loading Any Frozen Pipeline & Data
```python
import json
import joblib
import pandas as pd
from src.data.splits import load
from src.models.baselines import model_frame
from src.utils.io import config

cfg = config("config/base.yaml")
manifest = json.loads(open("artifacts/p3/model_manifest.json").read())
cols = manifest["feature_order"]  # exact 50 keep fields

frames = load(cfg, names=("train", "val"))  # omit "test" during development
X_val = model_frame(frames["val"][cols])

# Load Champion (stack_lr) or standalone booster (catboost / xgb_hist / lgbm_native_flags)
champ = joblib.load(manifest["models"]["stack_lr"]["path"])
p_val = champ.predict_proba(X_val)[:, 1]
```

---

## 3. Specific Notes for P4 Deliverables

1. **Calibration & Cost-Sensitive Thresholding (Step 10):**
   - Compare Platt scaling (`LogisticRegression` on logits) vs isotonic regression fitted **on Validation predictions** (`*_val.parquet`) for `stack_lr`, `catboost`, `xgb_hist`, `lgbm_native_flags`, `adaboost` (`Test ECE = 0.1099`, `artifacts/p3/tables/benchmark.csv`), and `catboost_imb_class_weights` (`Test ECE = 0.2390`, `artifacts/p3/tables/benchmark.csv`).
   - Verify two empirical patterns from `artifacts/p3/tables/benchmark.csv` and `artifacts/p3/tables/imbalance_study.csv`: (a) `stack_lr` already has `Test ECE = 0.0045` (`Test Brier = 0.1386`, `calibration_slope = 1.010`) because its forward-chained OOF meta-learner absorbed upward default-rate shift (`Train 17.69% → Val 20.20% / Test 19.91%`), whereas standalone boosters (`catboost`, `xgb_hist`, `lgbm_native_flags`) have `Test ECE = 0.0242–0.0291` and positive calibration intercepts (`+0.276` to `+0.343`); and (b) recalibrating `catboost_imb_class_weights` (`Test Brier = 0.2013`, `val_threshold_10 = 0.3121`) removes the $+\ln(n_-/n_+)$ prior-odds offset and restores calibrated cost cutoffs.
   - Re-use `src.evaluation.cost_sensitive.select` and `policies` (`cost_ratios = [5.0, 10.0, 15.0]`).
2. **Analysis A (Feature Ablation):**
   - Use `FEATURE_GROUPS` in `src/features/block2.py` (`lc_grade`, `affordability`, `application`, `bureau`, `engineered`) and `extract_keep_flags` in `src/block3.py`. Drop raw columns *before* `UnderwritingFeatures` so dependent ratios (`loan_to_income`, etc.) drop cleanly. Use `xgb_hist` (`13.44s` refit in `artifacts/p3/tables/efficiency.csv`) or `lgbm_native_flags` (`14.80s` refit) for group-ablation refits on `Train` evaluated on `Val`.
3. **Analysis D (Sensitivity & Temporal Stability):**
   - `frames["val"]` and `frames["test"]` carry `issue_d` for monthly vintage slices, and `X_val` supports missingness-injection, Gaussian-noise, and rate-shock perturbations across `logreg_baseline`, `xgb_hist`, `catboost`, and `lgbm_monotone` (`artifacts/p3/tables/monotone_comparison.csv`).
4. **Analysis F (Interpretability & SHAP):**
   - For `shap.TreeExplainer`, use standalone **`xgb_hist`**, **`catboost`**, **`lgbm_native_flags`**, and **`lgbm_monotone`** (unwrap `pipe.named_steps["model"]` on `pipe.named_steps["preprocess"].transform(X_sample)` with `pipe.named_steps["preprocess"].get_feature_names_out()`). Because `stack_lr` (`artifacts/p3/models/stack_lr.joblib`) wraps 3 pipelines (`StackedEnsemble.members` + `meta`), report its OOF meta-weights from `artifacts/p3/stack/stack_lr_meta.json` (`0.6514` `catboost`, `0.2554` `logreg_baseline`, `0.1811` `xgb_hist`, intercept `+0.2685`) alongside SHAP explanations of its standalone tree members.

---

## 4. Prior Block 1 & Block 2 Cohort Provenance (Preserved)

- **Cohort & Splits (`splits/p1p2_300k/`):** `210,000` Train (`2007-06` to `2015-05`), `45,000` Val (`2015-05` to `2015-09`), `45,000` Test (`2015-09` to `2015-12`), `300,000` distinct borrowers verified via `audit.parquet` (`member_id` from `lc_loan.csv.zip`) and `excluded_loan_ids.parquet`.
- **Processed Inputs:** `data/processed/p1p2_300k/sample.parquet` (`sha256: f0d05645…adbf`), `data/processed/p1p2_300k/audit.parquet`, `artifacts/p1p2_300k/feature_audit.csv` (50 `keep` origination fields), `artifacts/p1p2_300k/missing_value_table.csv`.
- **P1/P2 Saved Pipelines:** 4 baselines in `artifacts/p1p2_300k/models/` (`constant`, `grade_only`, `logreg_baseline`, `woe_scorecard`) and 17 classical/PCA pipelines in `artifacts/p1p2_300k/block2/models/`.
- **Prosper Transfer Prep (for P5):** `data/processed/p1p2_300k/prosper_common.parquet` and `artifacts/p1p2_300k/tables/prosper_feature_mapping.csv` (17 mapped underwriting fields).
