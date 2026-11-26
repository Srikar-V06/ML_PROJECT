# Block 3 (P3) Handover Check & Environment Verification

Generated: Thu 26 Nov 2026 · Branch: `block3/boosters` (from `2f9572e`)

## 1. Environment Status (T0)

| Item | Status | Details |
|---|---|---|
| Python | Ready | `3.12.16` in `.venv/` via `uv` (no system Python changes) |
| Dependencies | Ready | All 29 pinned packages from `requirements.txt` + `imbalanced-learn==0.14.2` |
| macOS OpenMP | Ready | `libomp 23.1.6` installed via Homebrew for LightGBM/XGBoost |
| `.gitignore` | Updated | Un-ignored `artifacts/p3/*.csv`, `artifacts/p3/tables/*.csv`, `artifacts/p3/predictions/*.parquet`, `reports/tables/*.csv`, `reports/preds/*.parquet`; ignored `artifacts/p3/search.sqlite3*` |
| `pytest` | **21 / 21 passed** | Ran `.venv/bin/python -m pytest -v` (18.54s), including `test_p1p2_handoff` |

## 2. Dataset & Artifact Availability (B1)

| Artifact / File | Status | Notes |
|---|---|---|
| `splits/p1p2_300k/train_ids.parquet` | Verified (`MATCH`) | 210,000 IDs |
| `splits/p1p2_300k/val_ids.parquet` | Verified (`MATCH`) | 45,000 IDs |
| `splits/p1p2_300k/test_ids.parquet` | Verified (`MATCH`) | 45,000 IDs (untouched) |
| `splits/p1p2_300k/train_expanding_cv.parquet` | Verified (`MATCH`) | 4 expanding folds |
| `data/processed/p1p2_300k/sample.parquet` | **Missing** | Expected SHA-256 `f0d0564560252aacb95127afbb0be900e94c64acba15fc9557eae5f27d4eadbf` |
| `data/processed/p1p2_300k/audit.parquet` | **Missing** | Needed for borrower verification if rebuilding |
| Raw archives (`accepted_2007_to_2018Q4.csv.gz`, `lc_loan.csv.zip`) | **Not found** | `data/` directory absent on disk; running on 500-row synthetic fixture until data arrives |
| Saved P1/P2 models (21 `.joblib` files) | **Verified load & score** | 4 in `artifacts/p1p2_300k/models/`, 17 in `artifacts/p1p2_300k/block2/models/`; all 21 load cleanly and output valid `[0, 1]` probabilities on the synthetic fixture. Real 45k Val re-scoring deferred until `sample.parquet` is present. |
| Committed Val predictions (21 `*_val.parquet`) | **Aligned** | All 21 files share identical 45,000 `loan_id`s and `y_true` labels; 0 `*_test.parquet` files |

## 3. Optuna Triage Table (P2 `search_*.json` -> Trial 0 Warm Starts)

All 5 booster families completed 3 Optuna trials on up to 3,000 fit rows per expanding fold (`artifacts/p1p2_300k/block2/search_*.json`). Local `search.sqlite3` is absent, so P3 will use these `params` only as **Trial 0 warm starts** in a wider Optuna search (T3).

| Family | Status | Trials | Fit Row Limit | Mean Inner AUC | P2 Best Params (Trial 0 Warm Start) |
|---|---|---|---|---|---|
| `catboost` | Lightweight | 3 | 3,000 | 0.6984 | `{"iterations": 150, "learning_rate": 0.028645, "depth": 2}` |
| `xgboost` | Lightweight | 3 | 3,000 | 0.6954 | `{"n_estimators": 150, "learning_rate": 0.028645, "max_depth": 2}` |
| `gradient_boosting` | Lightweight | 3 | 3,000 | 0.6936 | `{"n_estimators": 150, "learning_rate": 0.028645, "max_depth": 2}` |
| `adaboost` | Lightweight | 3 | 3,000 | 0.6933 | `{"n_estimators": 200, "learning_rate": 0.208033}` |
| `lightgbm` | Lightweight | 3 | 3,000 | 0.6800 | `{"n_estimators": 150, "learning_rate": 0.028645, "num_leaves": 16}` |
| `hgb` (`HistGradientBoostingClassifier`) | Missing (not in P2) | 0 | — | — | Will be tuned from scratch in T3 |

*(Reference: P1 `logreg_baseline` Val ROC-AUC is `0.7473`, PR-AUC `0.4307`.)*

## 4. Booster Smoke Timing Check (500-Row Synthetic Fixture)

Because `sample.parquet` is not yet on disk, smoke timings ran on the prepared 500-row synthetic fixture (`308` Train rows, `66` Val rows, `50` features, `24` missing indicators, `threadpool_limits=2`).

| Model | Preprocessor | Fit Time (s) | Predict Time (s) | Fixture Val ROC-AUC | Fixture Val PR-AUC |
|---|---|---:|---:|---:|---:|
| `lightgbm` | `tree` (native NaN + flags) | 0.044 | 0.0041 | 0.6232 | 0.1946 |
| `xgboost` | `tree` (native NaN + flags) | 0.049 | 0.0031 | 0.6768 | 0.2199 |
| `hgb` | `tree` (native NaN + flags) | 0.103 | 0.0040 | 0.6196 | 0.2006 |
| `catboost` | `tree` (one-hot + flags) | 2.860 | 0.0039 | 0.6321 | 0.2301 |
| `gradient_boosting` | `tree` (`impute_tree=True`) | 0.253 | 0.0035 | 0.6911 | 0.2741 |
| `adaboost` | `tree` (`impute_tree=True`) | 0.065 | 0.0048 | 0.5982 | 0.2299 |
