# P4 Guide: what P3 did and what P4 has to do

This guide is based on `HANDOFF.md`, `artifacts/p3/report_block3.md`, `docs/interfaces.md`, and the repo itself, checked on 2026-11-27. All 26 models in the manifest have their `.joblib`, Val predictions and Test predictions on disk.

---

## Part 1: What P3 did

### The project so far
- **P1 (Block 1):** built the dataset. It has 300,000 Lending Club borrowers and 50 origination-time features. The data is split by time:
  - **Train:** 210,000 loans, 2007-06 to 2015-05, 17.69% default.
  - **Validation:** 45,000 loans, 2015-05 to 2015-09, 20.20% default.
  - **Test:** 45,000 loans, 2015-09 to 2015-12, 19.91% default.

  P1 also trained 4 baselines: `constant`, `grade_only`, `logreg_baseline` and `woe_scorecard`.
- **P2 (Block 2):** engineered 4 ratio features (`loan_to_income`, `payment_to_income`, `revolving_to_income`, `open_account_share`). It trained a decision tree, a random forest and PCA variants, and ran short booster hyperparameter searches.
- **P3 (Block 3):** did the boosting work and produced the final, one-time Test benchmark.

### What P3 built (Block 3)

| # | Work | What it means | Where it is |
|---|---|---|---|
| 1 | **Boosters** | Tuned XGBoost, CatBoost, LightGBM, HistGradientBoosting, classic GradientBoosting and AdaBoost. Tuning used Optuna on 4 expanding time-folds of Train. Early stopping used the **last 10% of Train**, never Val, so Val stays clean. | `src/block3.py`, `artifacts/p3/tables/t3_val_summary.csv`, `artifacts/p3/tables/recovery.csv` |
| 2 | **LightGBM missing-value test** | Compared native NaN handling, native plus missing flags, and median imputation plus flags. The difference is tiny (< 0.0005 AUC). | `artifacts/p3/tables/lgbm_arms_comparison.csv` |
| 3 | **Class-imbalance study** | Tried 4 options on XGBoost and CatBoost: none, class weights, undersampling and SMOTE. **Finding:** class weights and undersampling don't improve ranking but **wreck calibration** (ECE 0.02 → 0.24, mean predicted probability ≈ 0.45 against a real rate of 0.20). SMOTE slightly hurts. Best approach: train with no rebalancing, then calibrate and pick a cost threshold. | `artifacts/p3/tables/imbalance_study.csv`, `artifacts/p3/plots/imbalance_study.png` |
| 4 | **Monotone LightGBM** | Forced 7 features to move in the "sensible" direction: higher `dti`, `int_rate`, `revol_util`, `loan_to_income` and `payment_to_income` mean more risk; higher `fico_range_low` and `annual_inc` mean less. Cost: −0.0010 ROC-AUC. The ICE audit shows that raw `annual_inc` is still not monotone end-to-end, because it also feeds the unconstrained `revolving_to_income`. | `artifacts/p3/tables/monotone_sign_check.md`, `artifacts/p3/tables/lgbm_monotone_ice_check.csv`, `artifacts/p3/plots/lgbm_monotone_ice.png` |
| 5 | **Ensembles** | `avg_top3` is a plain average of `catboost`, `xgb_hist` and `logreg_baseline`. **`stack_lr`** is a logistic-regression meta-model trained on out-of-time predictions; its weights are catboost 0.65, logreg 0.26 and xgb 0.18. | `artifacts/p3/stack/stack_lr_meta.json`, `artifacts/p3/tables/ensembles_val.csv` |
| 6 | **Freeze + one-shot Test** | Before touching Test, P3 froze all 26 models: SHA-256 hashes, the exact 50-column order, and cost thresholds for 5:1, 10:1 and 15:1 chosen **on Val**. It then scored Test **once**, with 1,000 bootstrap resamples for confidence intervals, and wrote a lock file. | `artifacts/p3/model_manifest.json`, `artifacts/p3/test_lock.json`, `artifacts/p3/tables/benchmark.csv` |
| 7 | **Significance tests** | DeLong tests and paired bootstrap tests with Holm correction: Champion vs all 25 other models, plus 7 pre-registered pairs. | `artifacts/p3/tables/significance.csv` |
| 8 | **Analysis C (efficiency)** | Training time, single-thread inference ms per 1,000 rows, memory, model size and tree counts for all 26 models, plus Pareto-frontier plots. | `artifacts/p3/tables/efficiency.csv`, `artifacts/p3/tables/pareto_frontier.csv`, `artifacts/p3/plots/pareto_*.png` |
| 9 | **Write-up** | Block 3 report sections, 3 slide outlines and viva Q&A. | `artifacts/p3/report_block3.md`, `docs/viva_qa.md` |

### Headline results (Test, 45k borrowers)

| Model | ROC-AUC | PR-AUC | Brier | ECE | Note |
|---|---:|---:|---:|---:|---|
| **`stack_lr` (Champion)** | 0.7439 | **0.4279** | **0.1386** | **0.0045** | Already well calibrated |
| `catboost` | **0.7442** | 0.4276 | 0.1395 | 0.0242 | Best single model |
| `xgb_hist` | 0.7429 | 0.4264 | 0.1402 | 0.0291 | Fast; recommended for real-time use |
| `lgbm_native_flags` | 0.7425 | 0.4243 | 0.1403 | 0.0279 | |
| `lgbm_monotone` | 0.7414 | 0.4235 | 0.1407 | 0.0323 | Explainable directions |
| `logreg_baseline` | 0.7372 | 0.4157 | 0.1406 | 0.0185 | P1 baseline |
| `adaboost` | 0.7364 | 0.4177 | 0.1538 | 0.1099 | Badly calibrated, probabilities squashed toward 0.5 |
| `catboost_imb_class_weights` | 0.7442 | 0.4279 | 0.2013 | 0.2390 | Very badly calibrated, probabilities about 2× too high |

**In short:** boosters beat the baselines by about +0.007 ROC-AUC and +0.012 PR-AUC, and the differences are significant. The models differ much more in **calibration** than in ranking, and P4 starts from that.

---

## Part 2: What P4 must do (Block 4)

Your scope is **Step 10 (calibration and cost-sensitive thresholds)** plus **Analyses A, D and F**. P5 appears to own the Prosper transfer study, since P1 already prepared `prosper_common.parquet` for it.

### Hard rules (from P3's handoff)
1. **Do NOT delete `artifacts/p3/test_lock.json` and do NOT re-run `src/final_benchmark.py`.** Test was used once and must stay that way.
2. **Do NOT touch `artifacts/p1p2_300k/predictions/`.** A test checks that it holds exactly 21 `*_val` files and 0 `*_test` files.
3. **Fit everything on Train or Val. Use Test only as a final read-out.** For calibration, use the Test prediction parquets that already exist; never re-predict Test.
4. **Write all outputs to `artifacts/p4/`** (tables in `artifacts/p4/tables/`, plots in `artifacts/p4/plots/`).
5. Keep `pytest` passing, because CI runs it on every push.

### Step 0: Setup (do this first; several things are broken locally)
- **LightGBM can't load (`Library not loaded: @rpath/libomp.dylib`):** on macOS, LightGBM needs the OpenMP runtime. Install it once:
  ```bash
  brew install libomp
  ```
- **Data path mismatch:** `config/base.yaml` expects `data/processed/p1p2_300k/...`, but your files are in **`data/preprocessed/p1p2_300k/`**. Without a fix, `load()` will fail. Fix it locally without changing the config everyone shares:
  ```bash
  cd ML_PROJECT && ln -s preprocessed data/processed
  ```
- **`shap` isn't installed** in `../.venv` and isn't in `requirements.txt`. Install it and add the pinned version to `requirements.txt` so CI and teammates match:
  ```bash
  ../.venv/bin/pip install shap
  ```
- Your venv runs Python 3.11; the README says 3.12. That works for now, but keep it in mind if a package complains.
- Work on a branch: `git checkout -b block4/calibration-analyses`.
- Check the baseline first: `../.venv/bin/python -m pytest -q`.

### Step 10: Calibration and cost-sensitive thresholds (main deliverable)

**Goal:** show that Platt scaling or isotonic regression, fitted on **Val**, fixes the badly calibrated models, and measure how much that saves in money terms.

**Models:** `stack_lr`, `catboost`, `xgb_hist`, `lgbm_native_flags`, `adaboost` and `catboost_imb_class_weights`.

**Procedure for each model:**
1. Load `artifacts/p3/predictions/<name>_val.parquet` and `<name>_test.parquet`. They have the columns `loan_id`, `y_true` and `y_prob`. Check that the `loan_id` values line up.
2. Fit two calibrators on the **Val** predictions:
   - **Platt:** `LogisticRegression` on `logit(y_prob)`.
   - **Isotonic:** `IsotonicRegression(out_of_bounds="clip")`.
3. Pick new 5:1, 10:1 and 15:1 thresholds on the *calibrated* Val probabilities with `src.evaluation.cost_sensitive.select`.
4. Transform the Test probabilities **once**. Report raw, Platt and isotonic results side by side: Brier, ECE, log-loss, calibration slope and intercept, and cost per 1,000 applicants at each ratio (use `policies`). ROC-AUC should not change (Platt) or should barely change (isotonic, because of ties). Mention this in the write-up.
5. Make reliability diagrams (raw vs calibrated) and a cost-curve plot.

**Honesty note:** if you fit the calibrator and pick the threshold on the same Val data, Val results look slightly optimistic. Better: run 5-fold cross-fitting on Val to report Val metrics, then refit on all of Val before transforming Test. Mention this in the report.

**Things to confirm (P3's hypotheses):**
- (a) `stack_lr` is already calibrated (ECE 0.0045, slope 1.01). Calibration should help it very little. The single boosters have a positive intercept (+0.28 to +0.34) because the default rate rose from 17.7% (Train) to about 20% (Val and Test), so calibration should fix them.
- (b) `catboost_imb_class_weights` is shifted by +ln(n₋/n₊) ≈ +1.54 in log-odds. Calibration should bring Brier back from 0.2013 to about 0.14 and move its 10:1 threshold from 0.31 to about 0.09–0.10.

**Code skeleton:**
```python
import numpy as np, pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from src.evaluation.cost_sensitive import select, policies
from src.evaluation.metrics import evaluate

def load(name, split):
    return pd.read_parquet(f"artifacts/p3/predictions/{name}_{split}.parquet").set_index("loan_id").sort_index()

for name in ["stack_lr", "catboost", "xgb_hist", "lgbm_native_flags", "adaboost", "catboost_imb_class_weights"]:
    v, t = load(name, "val"), load(name, "test")
    z = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    platt = LogisticRegression(C=1e6).fit(z(v.y_prob.values), v.y_true)
    iso = IsotonicRegression(out_of_bounds="clip").fit(v.y_prob, v.y_true)
    for method, f in {"raw": lambda p: p,
                      "platt": lambda p: platt.predict_proba(z(p))[:, 1],
                      "isotonic": iso.predict}.items():
        pv, pt = f(v.y_prob.values), f(t.y_prob.values)
        thr = select(v.y_true, pv, [5.0, 10.0, 15.0])           # thresholds from Val only
        res = evaluate(t.y_true, pt, "test", f"{name}_{method}", "artifacts/p4/calibration",
                       val_thresholds=thr, n_bootstrap=1000, seed=42)
        # collect res + policies(t.y_true, pt, thr) into artifacts/p4/tables/calibration.csv
```

**Deliverables:** `artifacts/p4/tables/calibration.csv`, `artifacts/p4/tables/cost_savings.csv`, `artifacts/p4/plots/reliability_*.png`, `artifacts/p4/plots/cost_curves.png`.

### Analysis A: Feature-group ablation

**Question:** how much does each group of features contribute?

- The groups come from `FEATURE_GROUPS` in `src/features/block2.py`: `lc_grade`, `affordability`, `application`, `bureau` and `engineered`.
- For each group, run **leave-one-group-out**: refit `xgb_hist` on Train without that group and evaluate on **Val only** (never Test). `xgb_hist` refits in about 13 s; `lgbm_native_flags` takes about 15 s.
- **Gotcha:** drop the raw columns **before** `UnderwritingFeatures` builds the ratios. If you drop `annual_inc` but `loan_to_income` is still built from it, the ablation is wrong. Rebuild the missing-flag list with `extract_keep_flags(...)` (in `src/block3.py`) restricted to the remaining columns.
- Reuse the tuned hyperparameters stored in the manifest or model so that only the features change.
- Optional extras: "only this group" runs, and a version with `lc_grade` removed. LC's own grade summarizes much of the risk signal, so it's interesting to see how much the bureau data recovers without it.
- Report ΔROC-AUC and ΔPR-AUC vs the full model, with paired bootstrap CIs from `src.evaluation.comparison` / `stats_tests`.
- **Deliverables:** `artifacts/p4/tables/ablation.csv` and a bar chart.

### Analysis D: Sensitivity and temporal stability

**1. Temporal stability (vintage drift).**
- Join the existing `*_val` and `*_test` prediction parquets with `issue_d` from `load(cfg, ["val", "test"])`. Slicing **existing** Test predictions is fine; you are not re-scoring Test.
- Compute ROC-AUC, PR-AUC, ECE and the actual vs predicted default rate per month for `logreg_baseline`, `xgb_hist`, `catboost`, `lgbm_monotone` and `stack_lr`.
- Optionally add PSI (population stability index) of key features from Train to Val to Test.

**2. Perturbation robustness**, on `X_val` with the frozen models:
- **Missingness injection:** set 5%, 10% and 20% of values in key features to NaN.
- **Gaussian noise:** add noise of 0.05σ, 0.1σ and 0.25σ to the numeric features.
- **Rate shock:** shift `int_rate` by +1, +2 and +3 points. Optionally shock `dti` too.
- Measure how AUC, the mean predicted probability and the approval rate at the 10:1 cutoff change.
- Compare monotone vs unconstrained models (`artifacts/p3/tables/monotone_comparison.csv`). The expectation is that monotone is more stable under shocks.

**Deliverables:** `artifacts/p4/tables/temporal_stability.csv`, `artifacts/p4/tables/sensitivity.csv`, and plots.

### Analysis F: Interpretability (SHAP)
- Use `shap.TreeExplainer` on `xgb_hist`, `catboost`, `lgbm_native_flags` and `lgbm_monotone`. Unwrap each pipeline:
  ```python
  pipe = joblib.load("artifacts/p3/models/xgb_hist.joblib")
  pre, model = pipe.named_steps["preprocess"], pipe.named_steps["model"]
  Xs = pre.transform(X_val.sample(5000, random_state=42))
  names = pre.get_feature_names_out()
  sv = shap.TreeExplainer(model).shap_values(Xs)
  ```
  For CatBoost you can also use its native `get_feature_importance(type="ShapValues")`.
- Make a **global** plot (beeswarm or bar) and a few **local** explanations, for example one approved and one rejected borderline borrower near the 10:1 cutoff.
- Make dependence plots for `int_rate`, `fico_range_low`, `dti` and `annual_inc`. Compare `lgbm_monotone` with `lgbm_native_flags`: does monotonicity show up in SHAP?
- Check whether the different models rank features the same way (Spearman correlation of mean |SHAP| across models).
- **`stack_lr` can't be passed straight to TreeExplainer**, because it wraps 3 pipelines. Report its meta-weights from `artifacts/p3/stack/stack_lr_meta.json` (catboost 0.6514, logreg 0.2554, xgb_hist 0.1811, intercept +0.2685) and explain it through the SHAP of its members.
- **Deliverables:** `artifacts/p4/tables/shap_importance.csv` and `artifacts/p4/plots/shap_*.png`.

### Finishing up (match P3's format)
1. Write `artifacts/p4/report_block4.md`: methodology, results tables with exact numbers and file paths, and findings for Step 10, A, D and F. Add about 3–4 slide outlines.
2. Add P4 questions and answers to `docs/viva_qa.md`.
3. Add tests in `tests/test_block4.py`. Examples: calibrators are fitted only on Val; the p1p2 predictions folder still has 21/0 files; Test predictions in `artifacts/p3/predictions` are unchanged (compare SHA-256 values).
4. Write a new `HANDOFF.md` section for P5.
5. Open a PR from your branch.

### Suggested order
Setup → **Step 10** (quickest, and it's the core) → **F** (SHAP) → **D** → **A** (most compute because of the refits) → report and tests.
