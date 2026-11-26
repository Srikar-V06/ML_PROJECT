# Block 3 Report Sections & Slide Outlines (Person 3 — Boosting, Imbalance, Ensembles, Final Benchmark & Efficiency)

---

## 1. Methodology & Leakage-Free Experimental Protocol

Block 3 inherits the frozen 300,000-borrower cohort from Blocks 1–2 (`210,000` Train: 2007-06 to 2015-05, default rate `17.69%`; `45,000` Validation: 2015-05 to 2015-09, `20.20%`; `45,000` Test: 2015-09 to 2015-12, `19.91%`), the 50 origination-time `keep` features, P2's 4 engineered underwriting ratios (`loan_to_income`, `payment_to_income`, `revolving_to_income`, `open_account_share`), and the 24 Train-derived missingness indicators (`ledger flags ∩ 50 keep fields`).

To prevent any validation or holdout leakage across tuning, early stopping, resampling, stacking, threshold selection, and final benchmarking, we enforced four structural safeguards in `src/block3.py`:

1. **Optuna Recovery & Expanding-Fold Tuning (`recovery.csv`):** P2's 3-trial, 3,000-row search JSONs (`mean CV AUC 0.680–0.698`) were enqueued as **Trial 0 warm starts** into widened Optuna TPE searches (`≤30` trials or `25-min` wall-clock cap per family, `25,000` seeded fit rows per fold) scored across all 4 chronological expanding Train folds (`splits/p1p2_300k/train_expanding_cv.parquet`). Preprocessors and missing flags (`0.5% < missing < 100%`) were fit strictly inside each fold's fitting partition.
2. **Chronological Train-Tail Early Stopping (`90% / 10%`):** Validation (`Val`) was never passed to `eval_set`. Instead, each booster was fit on the earliest 90% of Train (`189,000` borrowers) and early-stopped (`patience=50`; `staged_predict_proba` ROC-AUC for `HistGradientBoostingClassifier`, `GradientBoostingClassifier`, and `AdaBoostClassifier`, avoiding HGB's random internal split) on the latest chronological 10% Train tail (`21,000` borrowers). The final pipeline was then refit on 100% of Train using `round(best_iter / 0.9)` rounds.
3. **Pre-Test Candidate & Threshold Freeze (`model_manifest.json`):** All 26 candidate models (P1 baselines, P2 trees/forests + best Val PCA per classifier, and all P3 boosters, imbalance arms, monotone LightGBM, and ensembles) were evaluated on `Val` to lock cost-optimal decision thresholds (`5:1`, `10:1`, `15:1`), SHA-256 artifact digests, and the single Champion (`stack_lr`, highest Val PR-AUC `0.4410`).
4. **One-Shot Test Lock (`final_benchmark.py` & `test_lock.json`):** `verify_manifest_and_acquire_test_lock()` verified all 26 model SHA-256 digests and exact 50-column order, wrote `artifacts/p3/test_lock.json` (refusing any second execution), and scored the 45,000 Test borrowers once with **1,000-resample** percentile bootstrap 95% CIs (`n_bootstrap=1000`, overriding config's 250).

---

## 2. Boosting Families & Missing-Value Ablation

Every tuned gradient-boosted family substantially outperformed P1's penalized logistic baseline (`Test ROC 0.7372`, `PR 0.4157`), WoE scorecard (`ROC 0.7252`, `PR 0.3956`), and P2's random forest (`ROC 0.7340`, `PR 0.4080`).

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

- **Native categoricals matter:** `catboost` passing the 5 categoricals (`grade, sub_grade, home_ownership, verification_status, purpose`) natively with ordered target statistics and depth-7 oblivious trees achieved the highest standalone Test ROC-AUC (`0.7442`) and PR-AUC (`0.4276`), outperforming `xgb_hist` on ROC-AUC (`+0.0013`, DeLong `p_holm=0.0033`).
- **LightGBM 3-arm missingness study (`lgbm_arms_comparison.csv`):** Holding tuned hyperparameters fixed (`num_leaves=9, lr=0.0137, min_child_samples=162`), `lgbm_native_flags` (`Test ROC 0.7425`), `lgbm_native` (`0.7424`), and `lgbm_imputed_flags` (`0.7420`) performed within `0.0005` ROC-AUC of one another. Because LightGBM learns optimal default split directions for `NaN`s at every node, explicit median imputation + 24 binary flags is largely redundant for histogram GBDTs (unlike linear models), though native `NaN` handling edged out median imputation on both Val (`+0.0004` ROC) and Test (`+0.0005` ROC).
- **PCA hurts trees on Test too:** Confirming P2's validation hypothesis, orthogonal rotation degraded `pca_lightgbm_80` (`0.7100` ROC) and `pca_random_forest_99` (`0.7215` ROC) far more than `pca_lr_95` (`0.7349` vs `0.7372` full `logreg_baseline`), because axis-aligned splits cannot cleanly recover single bureau/pricing cutoffs after dense rotation.
- **AdaBoost calibration pathology:** While `adaboost` achieved respectable ranking (`Test ROC 0.7364`), exponential loss pushed predicted probabilities toward `0.5` (`calibration_slope = 2.54`, `Test ECE = 0.1099`, `Brier = 0.1538`).

---

## 3. Class Imbalance Study (`imbalance_study.csv`, `imbalance_study.png`)

On the top-2 booster families selected strictly by Val PR-AUC (`xgboost` and `catboost`), we compared four training-only treatments across all `210,000` Train borrowers: (1) `none`, (2) `class_weights` (`scale_pos_weight = 4.65` / `auto_class_weights="Balanced"`), (3) `RandomUnderSampler(1.0)` (`74,290` rows), and (4) `imblearn.pipeline.Pipeline` with `SMOTE` after median imputation + one-hot encoding (`345,710` rows).

| Family | Arm | Val PR-AUC | Val Mean $\hat{p}$ | Test ROC-AUC | Test PR-AUC | Test Brier | Test ECE | Cal Intercept |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `xgboost` | `none` | 0.4401 | 0.1801 | **0.7429** | **0.4264** | **0.1402** | 0.0291 | +0.343 |
| `xgboost` | `class_weights` | **0.4405** | 0.4490 | 0.7427 | 0.4262 | 0.1985 | 0.2340 | -1.309 |
| `xgboost` | `undersample` | 0.4367 | 0.4510 | 0.7418 | 0.4248 | 0.1996 | 0.2358 | -1.319 |
| `xgboost` | `smote` | 0.4350 | 0.1902 | 0.7418 | 0.4228 | 0.1398 | **0.0187** | +0.269 |
| `catboost` | `none` | 0.4400 | 0.1840 | **0.7442** | 0.4276 | 0.1395 | 0.0242 | +0.276 |
| `catboost` | `class_weights` | **0.4406** | 0.4516 | **0.7442** | **0.4279** | 0.2013 | 0.2390 | -1.342 |
| `catboost` | `undersample` | 0.4365 | 0.4567 | 0.7431 | 0.4256 | 0.2033 | 0.2440 | -1.359 |
| `catboost` | `smote` | 0.4359 | 0.1901 | 0.7438 | 0.4259 | **0.1394** | **0.0180** | +0.242 |

**Takeaways for lending decisions:**
1. **Weighting and undersampling wreck calibration without helping Test ranking:** Reweighting shifts prior log-odds by $+\ln(n_- / n_+) \approx +1.54$, inflating mean predicted default probability to `~0.45` against a `0.199` Test default rate (`Test Brier` jumps from `0.140` to `0.199–0.203`, `Test ECE` jumps `8–10×` to `0.234–0.244`). Applying an unadjusted Bayesian cost threshold $t^* = 1/(1+10) = 0.0909$ directly to a class-weighted booster would reject nearly the entire applicant pool!
2. **Undersampling discards 65% of repaid loans (`210k → 74k`),** hurting Test PR-AUC by `-0.0016` to `-0.0020`.
3. **Why Euclidean SMOTE did not inflate probabilities to 0.50:** Because plain SMOTE interpolates in `137`-dimensional one-hot + indicator space, synthetic minority points acquire fractional dummy values (`0 < x < 1`) that never occur in real borrowers (`x ∈ {0, 1}`). Shallow/medium GBDT splits isolate those off-manifold synthetic regions, leaving real out-of-time borrowers with `mean_p ≈ 0.190` while losing `-0.0017` to `-0.0036` Test PR-AUC and bloating serialized model size by `4–13×` (`5.35–6.58 MB`). **Conclusion:** Train on the natural `17.7%` prior (`none`), calibrate on Val (P4), and handle asymmetric `10:1` default costs via threshold optimization.

---

## 4. Monotone-Constraint LightGBM (`monotone_sign_check.md`, `lgbm_monotone_ice.png`)

All 7 domain-specified features passed our Train-only decile Spearman audit with `|ρ| ≥ 0.988` (`+1`: `dti`, `int_rate`, `revol_util`, `loan_to_income`, `payment_to_income`; `-1`: `fico_range_low`, `annual_inc`; note `fico_range_low` is the sole FICO variable in the 50-field schema). Enforcing `monotone_constraints` (`monotone_constraints_method="advanced"`) cost **only `-0.0010` Test ROC-AUC** (`0.7414` vs `0.7424` `lgbm_native`, DeLong `p_holm=0.0023`) and **`-0.0009` Test PR-AUC** (`0.4235` vs `0.4245`, `p_holm=0.51`, statistically indistinguishable), while shrinking tree complexity (`1,441` vs `1,764` trees) and inference time (`39.5` vs `45.0 ms/1k`).

Post-hoc ICE sweeps over `2,000` Val borrowers confirmed exact `0 / 2,000` model-input violations on 6 of 7 features (`70 / 2,000` tiny `≤5.8e-4` numerical steps on `annual_inc` under `advanced` heuristic pruning). Crucially, varying *raw* `annual_inc` also alters `loan_to_income`, `payment_to_income`, and the **unconstrained** `revolving_to_income`, demonstrating that credit-governance monotonicity guarantees require either constraining all coupled ratios or omitting redundant unconstrained ratios.

---

## 5. Ensembles & Statistical Significance (`ensembles_val.csv`, `significance.csv`)

Combining three diverse families — `xgb_hist` (depth-wise histogram GBDT), `catboost` (oblivious trees + native categoricals, Spearman `0.993` with XGB), and `logreg_baseline` (linear L2, Spearman `0.967` with both boosters) — improved both ranking and probability accuracy:
- **Simple probability average (`avg_top3`):** Zero extra training cost, achieving `Test ROC 0.7435` and `Test PR 0.4273` (beating `xgb_hist` and `lgbm_*`).
- **Forward-chained stacker (`stack_lr`, Champion):** Refitting all three base pipelines across the 4 expanding Train folds produced `168,000` strictly out-of-time out-of-fold predictions (`rows 42k–210k`, zero Val/Test leakage). Fitting `LogisticRegression(C=1.0)` on OOF logits learned positive weights on all three families (`0.6514` CatBoost + `0.2554` LogReg + `0.1811` XGBoost, intercept `+0.2685`). Because the OOF chunks span later vintages with rising default rates (`17.7% → 20.2%`), the meta-intercept absorbed temporal prior shift, giving `stack_lr` **the best calibrated probabilities in the entire 26-model benchmark** (`Test Brier = 0.1386`, `Test ECE = 0.0045`, `calibration_slope = 1.010`) alongside top-tied **Test PR-AUC (`0.4279`)**.

**Key Holm-adjusted Test significance results (`B = 1,000`):**
- `stack_lr` vs `logreg_baseline`: `+0.0067` ROC (`p_holm = 4.93e-22`), `+0.0122` PR (`p_holm = 0.025`).
- `stack_lr` vs `xgb_hist`: `+0.0011` ROC (`p_holm = 0.0146`), `-0.0016` Brier (`p_holm = 0.025`).
- `stack_lr` vs `avg_top3`: `+0.0005` ROC (`p_holm = 0.0029`), `-0.0012` Brier (`p_holm = 0.025`).
- `stack_lr` vs `catboost`: tied on discrimination (`-0.0002` ROC `p_holm = 0.94`, `+0.0003` PR `p_holm = 0.59`), while `stack_lr` significantly improves Brier (`-0.0009`, `p_holm = 0.025`) and cuts Test ECE from `0.0242` to `0.0045`.

---

## 6. Analysis C — Model Efficiency & Real-Time Lending Pareto Trade-Offs (`efficiency.csv`, `pareto_*.png`)

All 26 frozen models were profiled in isolated subprocesses on the same 10-core Apple Silicon machine (training: 4 threads, median of 3 refits if `<120s` else `n=1`; inference: **1 thread**, median of 3 runs on 1,000 Val borrowers after warm-up). Because Python's `tracemalloc` cannot observe C++ heap allocations inside LightGBM, XGBoost, and CatBoost (`infer_tracemalloc ≈ 2.6–3.0 MB` vs actual process `infer_rss ≈ 198–339 MB`), we report both metrics.

- **Inference-latency Pareto frontier (`pareto_infer_time.png`):** `grade_only` (`1.3 ms/1k`, `PR 0.3647`) → `logreg_baseline` (`5.7 ms/1k`, `5.3 KB`, `PR 0.4157`) → `xgb_hist` (`12.6 ms/1k`, `13.4s` train, `0.40 MB`, `1,893` depth-3 trees, `PR 0.4264`) → `catboost` (`19.4 ms/1k`, `126.7s` train, `1.55 MB`, `1,549` depth-7 oblivious trees, `PR 0.4276`).
- **Production recommendation:** For **high-throughput real-time applicant scoring** under strict latency SLAs (`<15 ms` per 1,000 applicants / `~13 µs` per applicant on 1 CPU thread), **`xgb_hist`** (or **`catboost`** at `19.4 ms/1k` if `+0.0012` PR-AUC / `+0.0013` ROC-AUC justifies `9.4×` slower retraining) is the Pareto-optimal standalone deployment choice, while **`stack_lr`** (`42.1 ms/1k` — still only `42 µs` per loan!) is ideal when batch/online credit underwriting prioritizes peak PR-AUC (`0.4279`) and near-perfect out-of-the-box calibration (`ECE = 0.0045`).

---

## 7. Presentation Slide Outlines (Block 3 — 3 Slides)

### Slide 1: Boosting Benchmark & Statistical Rigor (45,000 Held-Out Test Borrowers)
- **Visuals:** `artifacts/p3/plots/benchmark_pr_overlay.png` + `artifacts/p3/plots/benchmark_ci_bars.png`
- **Key bullets:**
  - All tuned boosters trained with leakage-free **90/10 chronological Train-tail early stopping** (`Val` untouched during fitting; `Test` scored once under cryptographic SHA-256 manifest lock).
  - Native-categorical **CatBoost** (`ROC 0.7442`, `PR 0.4276`) and **XGBoost `hist`** (`ROC 0.7429`, `PR 0.4264`) beat P1 Logistic (`0.7372` / `0.4157`, DeLong $p_{\text{Holm}} < 10^{-8}$) and P2 Random Forest (`0.7340` / `0.4080`).
  - **Forward-chained Stacker (`stack_lr`)** (`168k` OOT OOF rows) ties CatBoost on ranking (`PR 0.4279`) while achieving the best Test Brier (`0.1386`) and ECE (`0.0045`).

### Slide 2: Domain Governance — Class Imbalance Trap & Monotone Constraints
- **Visuals:** `artifacts/p3/plots/imbalance_study.png` + `artifacts/p3/plots/lgbm_monotone_ice.png`
- **Key bullets:**
  - **Imbalance lesson:** `class_weights` and `undersample` yield `≤ +0.0003` Test PR-AUC gain but **wreck probability calibration** (mean $\hat{p} \approx 0.45$ vs `0.199` base rate; Brier `0.140 → 0.201`; ECE `0.02 → 0.24`), breaking cost-based cutoffs; Euclidean `SMOTE` synthesizes off-manifold fractional one-hots (`-0.002` to `-0.004` PR-AUC).
  - **Monotone LightGBM (`lgbm_monotone`):** Enforcing 7 empirically verified credit signs (`+DTI, +int_rate, +revol_util, +LTI, +PTI; -FICO, -income`) costs only **`-0.0010` Test ROC-AUC** (`0.7414` vs `0.7424`) and **`-0.0009` PR-AUC** ($p_{\text{Holm}}=0.51$) while guaranteeing adverse-action directional sanity on model inputs.

### Slide 3: Analysis C — Compute Efficiency & Real-Time Scoring Pareto Frontier
- **Visuals:** `artifacts/p3/plots/pareto_infer_time.png` + `artifacts/p3/plots/pareto_train_time.png`
- **Key bullets:**
  - Unified single-machine profiling across all 26 models (`tracemalloc` + C++ process `RSS`, 1-thread latency, artifact size, tree/leaf complexity).
  - **Real-time Pareto frontier:** `logreg_baseline` (`5.7 ms/1k`) → `xgb_hist` (`12.6 ms/1k`, `13.4s` fit, `0.40 MB`) → `catboost` (`19.4 ms/1k`, `126.7s` fit, `1.55 MB`).
  - **Lender takeaway:** Deploy `xgb_hist` or `catboost` for ultra-low-latency APIs (`<20 µs`/loan); deploy `stack_lr` (`42 µs`/loan) when calibration (`ECE 0.45%`) and top PR-AUC (`0.4279`) govern automated cutoff policies.
