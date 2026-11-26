# Block 3 Report Sections & Slide Outlines (Person 3 — Boosting, Imbalance, Ensembles, Final Benchmark & Efficiency)

---

## 1. Methodology & Leakage-Free Experimental Protocol

Block 3 uses the frozen 300,000-borrower cohort verified in `artifacts/p3/tables/handover_check.md` (`210,000` Train: 2007-06 to 2015-05, default rate `17.69%`; `45,000` Validation: 2015-05 to 2015-09, `20.20%`; `45,000` Test: 2015-09 to 2015-12, `19.91%`), the 50 origination-time `keep` fields (`artifacts/p1p2_300k/feature_audit.csv`), P2's 4 engineered underwriting ratios (`loan_to_income`, `payment_to_income`, `revolving_to_income`, `open_account_share`), and the 24 Train-derived missingness indicators (`ledger flags ∩ 50 keep fields`).

Four code-level controls in `src/block3.py` prevent validation or holdout leakage:

1. **Optuna Recovery & Expanding-Fold Tuning (`artifacts/p3/tables/recovery.csv`):** P2's 3-trial, 3,000-row search JSONs (`mean CV AUC 0.6800–0.6984`, documented in `artifacts/p3/tables/handover_check.md`) were enqueued as **Trial 0 warm starts** into widened Optuna TPE searches (`≤30` trials or `25-min` cap per family, `25,000` seeded fit rows per fold) across the 4 expanding Train folds (`splits/p1p2_300k/train_expanding_cv.parquet`). Preprocessors and missing flags (`0.5% < missing < 100%`) were fit inside each fold's fitting partition only.
2. **Chronological Train-Tail Early Stopping (`artifacts/p3/tables/t3_val_summary.csv`):** `Val` was never passed to `eval_set`. Each booster was fit on the earliest 90% of Train (`189,000` borrowers) and early-stopped (`patience=50`; `staged_predict_proba` ROC-AUC for `hgb`, `gb_classic`, and `adaboost`, avoiding `HistGradientBoostingClassifier`'s random internal split) on the latest 10% chronological Train tail (`21,000` borrowers), then refit on 100% of Train with `round(best_iter / 0.9)` rounds.
3. **Pre-Test Candidate & Threshold Freeze (`artifacts/p3/model_manifest.json`, `artifacts/p3/tables/frozen_val_leaderboard.csv`):** All 26 candidate models were scored on `Val` to lock cost-optimal thresholds (`5:1`, `10:1`, `15:1`), SHA-256 digests, and the Val-selected Champion (`stack_lr`, Val PR-AUC `0.4410`, Val ROC-AUC `0.7529`).
4. **One-Shot Test Lock (`src/final_benchmark.py`, `artifacts/p3/test_lock.json`, `artifacts/p3/tables/benchmark.csv`):** `verify_manifest_and_acquire_test_lock()` verified all 26 model SHA-256 digests and exact 50-column order, wrote `artifacts/p3/test_lock.json`, and scored the 45,000 Test borrowers once with **1,000-resample** percentile bootstrap 95% CIs (`n_bootstrap=1000`, overriding config's 250).

---

## 2. Boosting Families & Missing-Value Ablation (`artifacts/p3/tables/benchmark.csv`)

On the 45,000-borrower Test cohort (`artifacts/p3/tables/benchmark.csv`), every full-Train GBDT outperformed P1's `logreg_baseline` (`Test ROC 0.7372`, `PR 0.4157`), `woe_scorecard` (`ROC 0.7252`, `PR 0.3956`), and P2's `random_forest` (`ROC 0.7340`, `PR 0.4080`).

| Model | Block | Fit Rows | Val PR-AUC | Test ROC-AUC (95% CI) | Test PR-AUC (95% CI) | Test Brier | DR @ 80% App |
|---|---|---:|---:|---|---|---:|---:|
| `stack_lr` *(Champion)* | P3 | 210,000 | **0.4410** | 0.7439 [0.7381, 0.7495] | **0.4279** [0.4172, 0.4381] | **0.1386** | 14.03% |
| `catboost` (native `cat_features`) | P3 | 210,000 | 0.4400 | **0.7442** [0.7384, 0.7498] | 0.4276 [0.4170, 0.4377] | 0.1395 | **14.02%** |
| `avg_top3` | P3 | 210,000 | 0.4409 | 0.7435 [0.7377, 0.7490] | 0.4273 [0.4169, 0.4377] | 0.1398 | 14.04% |
| `xgb_hist` (`tree_method="hist"`) | P3 | 210,000 | 0.4401 | 0.7429 [0.7370, 0.7486] | 0.4264 [0.4159, 0.4367] | 0.1402 | 14.07% |
| `lgbm_imputed_flags` | P3 | 210,000 | 0.4387 | 0.7420 [0.7361, 0.7476] | 0.4247 [0.4142, 0.4353] | 0.1404 | **13.99%** |
| `lgbm_native` | P3 | 210,000 | 0.4392 | 0.7424 [0.7365, 0.7480] | 0.4245 [0.4141, 0.4349] | 0.1403 | 14.00% |
| `lgbm_native_flags` | P3 | 210,000 | 0.4394 | 0.7425 [0.7367, 0.7481] | 0.4243 [0.4137, 0.4349] | 0.1403 | 14.00% |
| `lgbm_monotone` | P3 | 210,000 | 0.4366 | 0.7414 [0.7354, 0.7470] | 0.4235 [0.4130, 0.4337] | 0.1407 | 14.09% |
| `hgb` (`HistGradientBoosting`) | P3 | 210,000 | 0.4377 | 0.7418 [0.7360, 0.7475] | 0.4227 [0.4122, 0.4330] | 0.1402 | 14.06% |
| `gb_classic` *(50k stratified)* | P3 | 50,000 | 0.4310 | 0.7378 [0.7321, 0.7434] | 0.4200 [0.4094, 0.4302] | 0.1414 | 14.21% |
| `adaboost` (depth-1 base, full) | P3 | 210,000 | 0.4301 | 0.7364 [0.7304, 0.7420] | 0.4177 [0.4078, 0.4278] | 0.1538 | 14.18% |
| `logreg_baseline` | P1 | 210,000 | 0.4307 | 0.7372 [0.7316, 0.7426] | 0.4157 [0.4055, 0.4264] | 0.1406 | 14.24% |
| `pca_lr_95` *(20k subsample)* | P2 | 20,000 | 0.4260 | 0.7349 [0.7290, 0.7404] | 0.4106 [0.4002, 0.4215] | 0.1420 | 14.25% |
| `random_forest` | P2 | 210,000 | 0.4204 | 0.7340 [0.7281, 0.7394] | 0.4080 [0.3981, 0.4184] | 0.1449 | 14.30% |
| `woe_scorecard` | P1 | 210,000 | 0.4102 | 0.7252 [0.7194, 0.7307] | 0.3956 [0.3855, 0.4057] | 0.1431 | 14.66% |
| `pca_random_forest_99` *(20k)* | P2 | 20,000 | 0.4073 | 0.7215 [0.7158, 0.7270] | 0.3953 [0.3851, 0.4058] | 0.1474 | 14.45% |
| `pca_lightgbm_80` *(20k)* | P2 | 20,000 | 0.3949 | 0.7100 [0.7039, 0.7157] | 0.3784 [0.3686, 0.3893] | 0.1458 | 14.94% |
| `grade_only` | P1 | 210,000 | 0.3722 | 0.7132 [0.7073, 0.7188] | 0.3647 [0.3553, 0.3738] | 0.1467 | 15.02% |
| `decision_tree` | P2 | 210,000 | 0.3411 | 0.6931 [0.6875, 0.6989] | 0.3297 [0.3220, 0.3377] | 0.1489 | 15.06% |

- **Native categoricals (`artifacts/p3/tables/benchmark.csv`, `artifacts/p3/tables/significance.csv`):** `catboost` passing the 5 categoricals (`grade, sub_grade, home_ownership, verification_status, purpose`) via `cat_features` with ordered target statistics and depth-7 oblivious trees (`1,549` trees, `198,272` leaves) achieved the highest standalone Test ROC-AUC (`0.7442`) and PR-AUC (`0.4276`), beating `xgb_hist` (`0.7429`) by `+0.0013` ROC-AUC (DeLong `p_holm = 0.0033`).
- **LightGBM 3-arm missingness ablation (`artifacts/p3/tables/lgbm_arms_comparison.csv`, `artifacts/p3/tables/benchmark.csv`):** With fixed tuned parameters (`num_leaves=9, lr=0.0137, min_child_samples=162`), `lgbm_native_flags` (`Val PR 0.4394`, `Test ROC 0.7425`), `lgbm_native` (`Val PR 0.4392`, `Test ROC 0.7424`), and `lgbm_imputed_flags` (`Val PR 0.4387`, `Test ROC 0.7420`) fell within `0.0005` Test ROC-AUC. Native `NaN` routing outperformed median imputation on both Val (`+0.0004` ROC) and Test (`+0.0005` ROC), and explicit binary flags added `+0.0001` Test ROC over `lgbm_native`.
- **PCA rotation vs raw features (`artifacts/p3/tables/benchmark.csv`):** Orthogonal rotation reduced `pca_lightgbm_80` (`Test ROC 0.7100` vs `0.7425` `lgbm_native_flags`) and `pca_random_forest_99` (`0.7215` vs `0.7340` `random_forest`) more than `pca_lr_95` (`0.7349` vs `0.7372` `logreg_baseline`), because axis-aligned splits lose single-feature bureau and pricing cutoffs after dense rotation (note PCA pipelines also used `fit_rows=20,000`).
- **AdaBoost probability compression (`artifacts/p3/tables/benchmark.csv`):** `adaboost` reached `Test ROC 0.7364` and `Test PR 0.4177` (`372` depth-1 stumps), but exponential loss compressed scores toward `0.5` (`calibration_slope = 2.540`, `Test ECE = 0.1099`, `Test Brier = 0.1538`).

---

## 3. Class Imbalance Study (`artifacts/p3/tables/imbalance_study.csv`, `artifacts/p3/tables/benchmark.csv`)

On the top-2 booster families ranked by Val PR-AUC in `artifacts/p3/tables/t3_val_summary.csv` (`xgboost` `0.4401` and `catboost` `0.4400`), four Train-only treatments were evaluated across all `210,000` Train borrowers: (1) `none`, (2) `class_weights` (`scale_pos_weight = 4.6535` / `auto_class_weights="Balanced"`), (3) `RandomUnderSampler(1.0)` (`74,290` rows), and (4) `imblearn.pipeline.Pipeline` with `SMOTE` after median imputation + one-hot encoding (`345,710` rows).

| Family | Arm | Val PR-AUC | Val Mean $\hat{p}$ | Test ROC-AUC | Test PR-AUC | Test Brier | Test ECE | Cal Intercept |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `xgboost` | `none` (`xgb_hist`) | 0.4401 | 0.1801 | **0.7429** | **0.4264** | **0.1402** | 0.0291 | +0.343 |
| `xgboost` | `class_weights` | **0.4405** | 0.4490 | 0.7427 | 0.4262 | 0.1985 | 0.2340 | -1.309 |
| `xgboost` | `undersample` | 0.4367 | 0.4510 | 0.7418 | 0.4248 | 0.1996 | 0.2358 | -1.319 |
| `xgboost` | `smote` | 0.4350 | 0.1902 | 0.7418 | 0.4228 | 0.1398 | **0.0187** | +0.269 |
| `catboost` | `none` (`catboost`) | 0.4400 | 0.1840 | **0.7442** | 0.4276 | 0.1395 | 0.0242 | +0.276 |
| `catboost` | `class_weights` | **0.4406** | 0.4516 | **0.7442** | **0.4279** | 0.2013 | 0.2390 | -1.342 |
| `catboost` | `undersample` | 0.4365 | 0.4567 | 0.7431 | 0.4256 | 0.2033 | 0.2440 | -1.359 |
| `catboost` | `smote` | 0.4359 | 0.1901 | 0.7438 | 0.4259 | **0.1394** | **0.0180** | +0.242 |

**Findings (`artifacts/p3/tables/imbalance_study.csv`, `artifacts/p3/tables/benchmark.csv`):**
1. **Weighting and undersampling distort calibration without meaningful Test ranking gain:** Reweighting shifts prior log-odds by $+\ln(n_- / n_+) = +1.538$, pushing Val mean predicted probability from `0.1801–0.1840` to `0.4490–0.4516` (`prob_ratio = 2.22–2.24×` the `0.2020` Val base rate) and increasing Test Brier from `0.1395–0.1402` to `0.1985–0.2033` and Test ECE from `0.0242–0.0291` to `0.2340–0.2440`, while changing Test PR-AUC by `-0.0002` (`xgboost_imb_class_weights`) to `+0.0003` (`catboost_imb_class_weights`). Applying an uncalibrated 10:1 cost cutoff $t^* = 1/(1+10) = 0.0909$ to `xgboost_imb_class_weights` (whose Val-optimized 10:1 cutoff shifted to `0.3096`) would reject nearly all applicants.
2. **Undersampling drops 64.6% of training rows (`210,000 → 74,290`),** reducing Test PR-AUC by `-0.0016` (`xgboost_imb_undersample` `0.4248`) and `-0.0020` (`catboost_imb_undersample` `0.4256`).
3. **Euclidean SMOTE off-manifold behavior:** Plain `SMOTE` in `137`-column one-hot + indicator space synthesizes minority vectors with fractional dummy coordinates (`0 < x < 1`) absent in real `{0, 1}` borrower records. Tree splits isolate those synthetic coordinates, leaving real Val borrowers at `mean_pred_prob = 0.1901–0.1902`, reducing Test PR-AUC by `-0.0017` (`catboost_imb_smote` `0.4259`) to `-0.0036` (`xgboost_imb_smote` `0.4228`), and inflating artifact size from `0.396 MB` to `5.351 MB` (`xgboost`) and `1.552 MB` to `6.584 MB` (`catboost`, `artifacts/p3/tables/efficiency.csv`). Unweighted training (`none`) + Val calibration + cost-threshold selection dominates resampling.

---

## 4. Monotone-Constraint LightGBM (`artifacts/p3/tables/monotone_sign_check.md`, `artifacts/p3/tables/monotone_comparison.csv`)

All 7 domain-specified features passed the Train-only decile Spearman check in `artifacts/p3/tables/monotone_sign_check.md` (`monotone_sign_deciles.csv`) with `|ρ| ≥ 0.988`: `+1` on `dti` (`+1.000`), `int_rate` (`+1.000`), `revol_util` (`+0.988`), `loan_to_income` (`+1.000`), and `payment_to_income` (`+0.988`); `-1` on `fico_range_low` (`-1.000`, sole FICO field in the 50-field set) and `annual_inc` (`-1.000`). Enforcing `monotone_constraints` (`monotone_constraints_method="advanced"`) in `lgbm_monotone` cost **`-0.0010` Test ROC-AUC** (`0.7414` vs `0.7424` `lgbm_native`, DeLong `p_holm = 0.0023` in `artifacts/p3/tables/significance.csv`) and **`-0.0009` Test PR-AUC** (`0.4235` vs `0.4245`, paired bootstrap `p_holm = 0.5075`), while reducing tree count (`1,441` vs `1,764`) and 1-thread latency (`39.50` vs `45.00 ms/1k`, `artifacts/p3/tables/efficiency.csv`).

In the 2,000-row Val ICE audit (`artifacts/p3/tables/lgbm_monotone_ice_check.csv`, `artifacts/p3/plots/lgbm_monotone_ice.png`), 6 of 7 constrained model inputs had `0 / 2,000` violations (`max_violation = 0.00`), while `annual_inc` at the model-input level had `70 / 2,000` small numerical steps (`max_violation = 5.79e-4`). Varying *raw* `annual_inc` produced `1,770 / 2,000` non-monotone curves (`max_violation = 5.06e-2`) because raw `annual_inc` simultaneously moves `loan_to_income`, `payment_to_income`, and unconstrained `revolving_to_income` (`0`); end-to-end raw-income monotonicity requires constraining or dropping `revolving_to_income`.

---

## 5. Ensembles & Statistical Significance (`artifacts/p3/tables/ensembles_val.csv`, `artifacts/p3/tables/significance.csv`)

Combining `xgb_hist`, `catboost` (Val Spearman `0.993` with `xgb_hist`, `artifacts/p3/tables/avg_top3_config.json`), and `logreg_baseline` (Spearman `0.967` with both boosters) yielded:
- **Simple probability average (`avg_top3`):** `0.00s` extra fit time, `Val PR 0.4409` (`ensembles_val.csv`), `Test ROC 0.7435`, `Test PR 0.4273`, `Test Brier 0.1398` (`benchmark.csv`).
- **Forward-chained stacker (`stack_lr`, Champion):** Refitting all 3 base pipelines across the 4 expanding Train folds generated `168,000` out-of-time OOF predictions (`rows 42k–210k`, `artifacts/p3/stack/stack_lr_meta.json`). `LogisticRegression(C=1.0)` on OOF logits learned weights `0.6514` (`catboost`), `0.2554` (`logreg_baseline`), `0.1811` (`xgb_hist`), and intercept `+0.2685`. Because the OOF folds span later vintages with higher default rates (`17.69%` Train → `20.20%` Val / `19.91%` Test), the meta-intercept corrected prior shift, giving `stack_lr` the lowest **Test Brier (`0.1386`)** and **Test ECE (`0.0045`, `calibration_slope = 1.010`)** across all 26 models alongside **Test PR-AUC `0.4279`** and **Test ROC-AUC `0.7439`** (`benchmark.csv`).

**Holm-adjusted Test significance (`artifacts/p3/tables/significance.csv`, `B = 1,000`):**
- `stack_lr` vs `logreg_baseline`: `+0.0067` ROC (DeLong `p_holm = 4.93e-22`), `+0.0122` PR (`p_holm = 0.0250`).
- `stack_lr` vs `xgb_hist`: `+0.0011` ROC (DeLong `p_holm = 0.0146`), `-0.0016` Brier (`p_holm = 0.0250`).
- `stack_lr` vs `avg_top3`: `+0.0005` ROC (DeLong `p_holm = 0.0029`), `+0.0006` PR (`p_holm = 0.5075`).
- `stack_lr` vs `catboost`: `-0.0002` ROC (DeLong `p_holm = 0.9447`), `+0.0003` PR (`p_holm = 0.5884`), `-0.0009` Brier (`p_holm = 0.0250`), with Test ECE dropping from `0.0242` to `0.0045`.
- `lgbm_native_flags` vs `logreg_baseline`: `+0.0053` ROC (DeLong `p_holm = 5.16e-8`), `+0.0087` PR (`p_holm = 0.0070`).

---

## 6. Analysis C — Model Efficiency & Pareto Trade-Offs (`artifacts/p3/tables/efficiency.csv`, `artifacts/p3/tables/pareto_frontier.csv`)

All 26 frozen models were timed in isolated subprocesses on the same 10-core Apple Silicon machine (training: 4 threads, median of 3 refits if `<120s` else `n=1` flagged; inference: **1 thread**, median of 3 runs on 1,000 Val rows after warm-up; `artifacts/p3/tables/efficiency.csv`). Python `tracemalloc` misses native C++ allocations in LightGBM, XGBoost, and CatBoost (`infer_tracemalloc_mb = 2.60–2.97 MB` vs subprocess `infer_rss_mb = 218.9–339.3 MB`), so both are recorded.

- **Train-time Pareto frontier (`artifacts/p3/tables/pareto_frontier.csv`, `pareto_train_time.png`):** `avg_top3` (`0.00s` extra, `Test PR 0.4273`) → `catboost` (`126.65s`, `n=1`, `PR 0.4276`) → `catboost_imb_class_weights` (`129.82s`, `n=1`, `PR 0.4279`, uncalibrated `ECE 0.2390`).
- **Inference-latency Pareto frontier (`artifacts/p3/tables/pareto_frontier.csv`, `pareto_infer_time.png`):** `grade_only` (`1.28 ms/1k`, `PR 0.3647`) → `logreg_baseline` (`5.69 ms/1k`, `0.0051 MB`, `134` params, `PR 0.4157`) → `xgboost_imb_class_weights` (`12.28 ms/1k`, `PR 0.4262`, `ECE 0.2340`) → `xgb_hist` (`12.57 ms/1k`, `13.44s` train, `0.396 MB`, `1,893` depth-3 trees, `14,684` leaves, `PR 0.4264`, `ECE 0.0291`) → `catboost` (`19.44 ms/1k`, `126.65s` train, `1.552 MB`, `1,549` depth-7 trees, `198,272` leaves, `PR 0.4276`, `ECE 0.0242`) → `catboost_imb_class_weights` (`19.55 ms/1k`, `PR 0.4279`).
- **Deployment selection:** For single-thread online scoring under `<15 ms/1k` (`12.6 µs`/applicant) with fast retraining (`13.44s`), deploy **`xgb_hist`**; if `19.44 ms/1k` and `126.65s` retraining fit budget, **`catboost`** adds `+0.0012` PR-AUC and `+0.0013` ROC-AUC (`p_holm = 0.0033`). When probability calibration (`Test Brier 0.1386`, `ECE 0.0045`) and peak PR-AUC (`0.4279`) govern automated cutoffs at `42.08 ms/1k` (`42.1 µs`/applicant), deploy **`stack_lr`**.

---

## 7. Presentation Slide Outlines (Block 3 — 3 Slides)

### Slide 1: Boosting Benchmark & Statistical Tests (`benchmark.csv`, `significance.csv`)
- **Visuals:** `artifacts/p3/plots/benchmark_pr_overlay.png` + `artifacts/p3/plots/benchmark_ci_bars.png`
- **Key bullets:**
  - 90/10 chronological Train-tail early stopping (`round(best_iter / 0.9)`) kept `Val` uncontaminated; `Test` (`45,000` borrowers) scored once under SHA-256 lock (`artifacts/p3/test_lock.json`).
  - `catboost` (`Test ROC 0.7442`, `PR 0.4276`) and `xgb_hist` (`ROC 0.7429`, `PR 0.4264`) beat `logreg_baseline` (`0.7372` / `0.4157`, DeLong $p_{\text{Holm}} \le 5.16 \times 10^{-8}$) and `random_forest` (`0.7340` / `0.4080`).
  - Forward-chained `stack_lr` (`168,000` OOT OOF rows) achieved `Test PR 0.4279`, lowest `Test Brier 0.1386`, and `Test ECE 0.0045`.

### Slide 2: Imbalance Calibration Distortion & Monotone LightGBM (`imbalance_study.csv`, `monotone_comparison.csv`)
- **Visuals:** `artifacts/p3/plots/imbalance_study.png` + `artifacts/p3/plots/lgbm_monotone_ice.png`
- **Key bullets:**
  - `class_weights` and `undersample` shifted Test PR-AUC by `-0.0020` to `+0.0003` while inflating Val mean $\hat{p}$ to `0.449–0.457` (base rate `0.2020`), raising Test Brier from `0.1395–0.1402` to `0.1985–0.2033` and ECE to `0.234–0.244`; `SMOTE` lost `-0.0017` to `-0.0036` Test PR-AUC via off-manifold fractional dummies.
  - `lgbm_monotone` enforced 7 Train-verified signs (`|ρ| ≥ 0.988` in `monotone_sign_check.md`) at `-0.0010` Test ROC cost (`0.7414` vs `0.7424` `lgbm_native`, $p_{\text{Holm}} = 0.0023$) and `-0.0009` PR cost ($p_{\text{Holm}} = 0.5075$), with `0/2,000` model-input ICE violations on 6 features (`lgbm_monotone_ice_check.csv`).

### Slide 3: Analysis C — Compute Efficiency & Pareto Frontiers (`efficiency.csv`, `pareto_frontier.csv`)
- **Visuals:** `artifacts/p3/plots/pareto_infer_time.png` + `artifacts/p3/plots/pareto_train_time.png`
- **Key bullets:**
  - Subprocess-isolated timings (`train_seconds`, 1-thread `infer_ms_per_1000`), artifact size, `tracemalloc` + C++ `RSS`, and tree/leaf counts across all 26 models (`efficiency.csv`).
  - Calibrated inference frontier (`pareto_frontier.csv`): `logreg_baseline` (`5.69 ms/1k`, `PR 0.4157`) → `xgb_hist` (`12.57 ms/1k`, `13.44s` train, `0.396 MB`, `PR 0.4264`) → `catboost` (`19.44 ms/1k`, `126.65s` train, `1.552 MB`, `PR 0.4276`).
  - Recommendation: `xgb_hist` for `<15 ms/1k` real-time APIs; `stack_lr` (`42.08 ms/1k`, `ECE 0.0045`) for calibrated cutoff automation.
