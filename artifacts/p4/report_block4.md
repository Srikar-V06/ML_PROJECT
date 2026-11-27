# Block 4 Report Sections & Slide Outlines (Person 4 — Calibration, Ablation, Stability & Interpretability)

Block 4 extends the frozen Block 3 benchmark without changing it. `artifacts/p3/**`, `artifacts/p3/test_lock.json`, and `artifacts/p1p2_300k/predictions/` are read-only. Everything below is reproducible with `python -m src.block4 --stage {calibration,shap,temporal,sensitivity,ablation}`. All Block 4 outputs live in `artifacts/p4/`.

---

## 1. Protocol: Validation for every decision, Test read once

1. **Development on Validation only.** Calibrators, method selection, thresholds, SHAP, perturbations, and ablation refits used Train and Validation only. `src/block4.py::predictions()` raises an error on any `*_test.parquet` read while `artifacts/p4/calibration_lock.json` is absent, and `tests/test_block4.py::test_test_predictions_blocked_before_lock` enforced that guard throughout development.
2. **Pre-registered selection rule.** For each model, the method with the lowest **5-fold cross-fitted Validation Brier** among `raw`, `platt`, and `isotonic` was selected (ties go to the simpler method). Cross-fitting scores each Validation row with a calibrator that never saw that row, so Validation results are not judged in-sample.
3. **One-shot Test read-out.** `python -m src.block4 --stage calibration --final` first wrote `artifacts/p4/calibration_lock.json`. The lock records:
   - commit `a8d6532`, time-stamped `2026-11-27T13:25Z`;
   - the selected methods and every threshold;
   - SHA-256 digests of all 18 calibrators;
   - digests of P3's `test_lock.json`, the manifest, and the 6 Test prediction files.

   The script then transformed **P3's saved Test probabilities**; no model re-scored Test. A second run raises an error. `tests/test_block4.py::test_p4_lock_matches_test_predictions` re-verifies those digests on every test run.
4. **Reproduction controls.**
   - Re-scoring unperturbed Validation reproduced all 6 frozen prediction files (max |Δp| < 1e-6; `src/analysis/block4_sensitivity.py`).
   - The ablation `full` arm reproduced P3's `xgb_hist` exactly: 1,893 rounds, max |Δp| = **0.0** (`artifacts/p4/tables/ablation.csv`).
   - Every SHAP decomposition sums to the model's log-odds within 1.6e-6 (`artifacts/p4/tables/shap_additivity.csv`).

---

## 2. Step 10 — Post-hoc calibration & cost-sensitive thresholds

**Method** (`src/evaluation/calibration.py`):
- **Platt scaling:** a logistic regression on `logit(p)`.
- **Isotonic regression:** clipped to [1e-6, 1 − 1e-6] so log-loss stays finite.

Both maps are monotone, so re-tuning a threshold after calibrating gives almost the same accept/reject decisions. Calibration therefore pays off when cutoffs come from the **Bayes cost rule** $t^* = 1/(1+R)$, which is optimal only for calibrated probabilities (reject when $R\,p > 1-p$). Three policies are compared per cost ratio $R \in \{5, 10, 15\}$:
- **raw + Bayes:** what happens without calibration;
- **calibrated + Bayes:** the fix;
- **raw + Validation-tuned cutoff:** P3's frozen manifest thresholds, as the reference.

**Validation (5-fold cross-fitted, `artifacts/p4/tables/calibration_val.csv`)** selected Platt for 5 models and isotonic for `adaboost`.

**Test read-out (`artifacts/p4/tables/calibration_test.csv`, `artifacts/p4/tables/cost_savings.csv`; ΔBrier is a 1,000-resample paired bootstrap):**

| Model | Method | ECE raw → cal | Cal. intercept raw → cal | Brier raw → cal | ΔBrier 95% CI | Bayes cost/1k @5:1 raw → cal |
|---|---|---|---|---|---|---|
| `catboost` | Platt | 0.0242 → **0.0084** | +0.276 → +0.009 | 0.1395 → 0.1386 | [−0.0011, −0.0006] | 573.8 → 572.4 |
| `xgb_hist` | Platt | 0.0291 → **0.0094** | +0.343 → +0.016 | 0.1402 → 0.1389 | [−0.0016, −0.0010] | 581.5 → 575.8 |
| `lgbm_native_flags` | Platt | 0.0279 → **0.0091** | +0.341 → +0.012 | 0.1403 → 0.1390 | [−0.0015, −0.0009] | 583.2 → 573.1 |
| `adaboost` | Isotonic | 0.1099 → **0.0132** | +0.638 → +0.023 | 0.1538 → 0.1402 | [−0.0145, −0.0129] | **778.4 → 590.4** |
| `catboost_imb_class_weights` | Platt | 0.2390 → **0.0073** | −1.342 → +0.008 | 0.2013 → **0.1387** | [−0.0646, −0.0607] | **717.8 → 570.1** |
| `stack_lr` *(Champion)* | Platt | 0.0045 → 0.0080 | +0.041 → +0.007 | 0.1386 → 0.1387 | [−0.00001, +0.00008] | 572.3 → 571.6 |

**Findings:**
1. **P3's hypothesis (a) holds.** The standalone boosters were biased toward too-low risk: positive intercepts of +0.28 to +0.34, and a Test mean $\hat p$ of 0.170–0.175 against a 0.199 base rate. That is the Train → Val/Test default-rate rise (17.69% → 20.20% / 19.91%). Platt removes the bias, cuts ECE 2.9–3.1×, and lowers Brier significantly for all three.
2. **P3's hypothesis (b) holds.** `catboost_imb_class_weights` carried a log-odds offset of −1.342 on Test, close to the predicted $-\ln(n_-/n_+) = -1.538$. Platt brings Brier from 0.2013 to 0.1387, level with the Champion. It also cuts the 5:1 Bayes-rule cost by **147.8 per 1,000 applicants** (87.4 at 10:1, 54.4 at 15:1). Uncalibrated, the Bayes cutoffs approved only 9% / 3% / 1% of Validation applicants at 5:1 / 10:1 / 15:1.
3. **AdaBoost becomes usable.** Its exponential loss squashed scores toward 0.5 (calibration slope 2.54). Raw, the Bayes rule approves almost no one (cost 778–801/1k). Isotonic brings cost to 590 / 708 / 749.
4. **Calibration makes the Bayes rule work as well as a tuned threshold.** After calibration, the Bayes rule lands within −6.3 to +2.8 cost/1k of P3's Validation-tuned cutoffs for every model. For well-ranked boosters the cost saving itself is small (≤ 10/1k), because their raw scores were already only mildly miscalibrated.
5. **Calibrating the Champion again did not help (reported as pre-registered).** `stack_lr`'s forward-chained meta-learner already absorbed the prior shift (Test ECE 0.0045). Platt won its Validation comparison by 0.00005 Brier, and on Test it left Brier unchanged (CI spans 0) while raising ECE to 0.0080. All Validation-fitted maps had Test slopes of about 0.95–0.96, because the Val → Test default rate moved from 20.20% to 19.91%. Recommendation: deploy `stack_lr` uncalibrated, and calibrate standalone boosters.
6. **Isotonic costs ranking.** Its step function merges ties and lowered Test PR-AUC by 0.0075–0.0117 on every model (ROC-AUC by up to 0.0006), versus 0 for Platt. That is why Platt is preferred at n = 45,000.

**Plots:** `artifacts/p4/plots/reliability_val_<model>.png`, `artifacts/p4/plots/reliability_test_<model>.png`.

---

## 3. Analysis A — Feature-group ablation (`artifacts/p4/tables/ablation.csv`, `artifacts/p4/plots/ablation.png`)

**Setup.**
- Model: leave-one-group-out refits of `xgb_hist` with P3's tuned parameters and P3's exact recipe (early stopping on the latest 10% of Train, then a refit on all 210,000 Train rows with `round(best_iter / 0.9)` rounds). Each arm re-learns its own round count; every arm is scored on Validation.
- Groups: `FEATURE_GROUPS` in `src/features/block2.py`.
- Dropping columns: raw columns are dropped *before* `UnderwritingFeatures`, which only builds a ratio when both of its inputs survive. Missing-value flags are restricted to the surviving columns.
- Statistics: ΔROC uses DeLong, ΔPR uses a 1,000-resample paired bootstrap, and both are Holm-adjusted over the 6 arms.

| Arm | Raw inputs | Model inputs | Rounds | Val ROC | Val PR | ΔROC (p_Holm) | ΔPR (p_Holm) |
|---|---:|---:|---:|---:|---:|---|---|
| `full` (= P3 `xgb_hist`) | 50 | 137 | 1,893 | 0.7517 | 0.4401 | — | — |
| `drop_affordability` | 44 | 128 | 1,761 | 0.7303 | 0.3977 | **−0.0214** (9.6e-56) | **−0.0424** (0.006) |
| `drop_bureau` | 12 | 74 | 1,486 | 0.7395 | 0.4159 | **−0.0122** (2.0e-28) | **−0.0242** (0.006) |
| `drop_lc_grade_and_int_rate` | 47 | 94 | 2,217 | 0.7452 | 0.4344 | **−0.0066** (1.6e-15) | **−0.0057** (0.006) |
| `drop_application` | 46 | 113 | 2,048 | 0.7496 | 0.4385 | −0.0021 (1.6e-4) | −0.0016 (0.36) |
| `drop_lc_grade` | 48 | 95 | 1,894 | 0.7514 | 0.4394 | −0.0003 (0.14) | −0.0007 (0.14) |
| `drop_engineered` | 50 | 133 | 2,049 | 0.7516 | 0.4397 | −0.0001 (0.75) | −0.0004 (0.59) |

**Findings:**
1. **Affordability fields carry the most signal.** These are `loan_amnt`, `term`, `installment`, `int_rate`, `annual_inc`, and `dti`; dropping them also removes 3 dependent ratios. Six fields are worth 1.75× the ROC-AUC of the 38-field bureau block.
2. **The grade is redundant with `int_rate`.** Dropping `grade` and `sub_grade` alone costs nothing significant (−0.0003) because Lending Club prices `int_rate` from `sub_grade`. Dropping all three costs 22× more (−0.0066). Lending Club's pricing is a strong risk summary that a booster can recover from either field.
3. **P2's 4 engineered ratios add nothing to a booster** (−0.0001, p = 0.75). Depth-3 trees approximate ratio boundaries from the raw fields. The ratios remain useful for linear models and for monotone constraints (Section 6).

---

## 4. Analysis D — Temporal stability & sensitivity

### 4a. Vintage stability (`artifacts/p4/tables/temporal_stability_{val,test}.csv`, `artifacts/p4/plots/temporal_stability_{val,test}.png`)

- **Ranking is stable.**
  - Validation (May–Sep 2015): ROC-AUC between 0.7330 and 0.7619.
  - Test (Sep–Dec 2015): ROC-AUC between 0.7275 and 0.7481.
  - The five tree-based and stacked models stay within 0.003–0.004 ROC-AUC of each other in every month, and `logreg_baseline` trails in every Validation and Test month.
- **Calibration drift is systematic for the standalone models.** On Test, the boosters under-predict in every month: mean $\hat p$ minus the observed rate is −0.020 to −0.037, against −0.016 to −0.022 for `logreg_baseline`. `stack_lr` stays within −0.0002 to −0.0079, with monthly Test ECE ≤ 0.0104 against 0.020–0.037 for the boosters. This is the month-level view of Step 10's finding (a).
- **Feature drift needs care with missing values** (`artifacts/p4/tables/feature_psi_{val,test}.csv`). Raw PSI from Train flags 18 bureau fields at PSI 0.2–1.4, but those fields were **not reported before 2012**: 100% missing in 2007–2011, 52% missing in 2012, and 0% from 2013. That is a reporting-regime change, not a change in the population. On observed values only (`psi_observed`), the single material drift is `int_rate`: 0.135 on Validation and **0.179** on Test (moderate, 0.1–0.25). All other fields are below 0.05.
- **Score drift is small** (`artifacts/p4/tables/score_psi_{val,test}.csv`): score PSI is 0.007–0.021 for all 6 models.

### 4b. Perturbation sensitivity (`artifacts/p4/tables/sensitivity.csv`, `artifacts/p4/plots/sensitivity.png`)

Frozen pipelines re-scored perturbed copies of all 45,000 Validation rows. No model was retrained. Approval rates use P3's frozen 10:1 cutoffs.

| Perturbation | Range | ΔROC-AUC | Approval effect |
|---|---|---|---|
| Missingness injection (numeric fields) | 5 / 10 / 20 % | `stack_lr` −0.005 / −0.010 / **−0.021** (most robust); `lgbm_monotone` up to **−0.055** | `xgb_hist` approval **falls 29.6% → 14.1%** at 20%; `lgbm_monotone` rises 28.1% → 33.2% |
| Gaussian noise (continuous fields, more than 10 levels) | 0.05 / 0.10 / 0.25 σ | ≤ −0.0075 at 0.25σ (`logreg_baseline` −0.0034) | within ±2 pp |
| `int_rate` shock, with `installment` re-priced | +1 / +2 / +3 pp | \|Δ\| ≤ 0.0015 | at +3 pp boosters approve 4–8 pp **fewer** (`stack_lr` 1.8 pp); `logreg_baseline` approves **more** (26.5% → 34.3%) |
| Same shock without re-pricing | +1 / +2 / +3 pp | same as re-priced | within 0.5 pp of the re-priced shock |
| Noise on `term` only | 0.05 σ | `xgb_hist` **−0.0120**; others ≤ 0.0005 | `xgb_hist` +1.6 pp |

**Findings:**
1. **Missing data is the dominant operational risk.** `xgb_hist` learned to route NaN as "pre-2012, riskier" (Section 4a), so a data-feed outage that blanks bureau fields would halve its approvals. `stack_lr`'s averaging over three differently-trained members halves the AUC loss.
2. **The logistic regression has a sign flip.** `logreg_baseline`'s standardized `int_rate` coefficient is **−0.32**, because `sub_grade` is in the same model: within a sub-grade, a higher rate is fitted as lower risk. A market-wide rate rise therefore *raises* its approvals, a governance failure that ROC-AUC does not reveal. Monotone constraints (`lgbm_monotone`) rule this out by construction.
3. **`xgb_hist` splits `term` at exactly 60.** XGBoost places all 340 `term` splits at `term < 60`, while LightGBM's 372 sit at the midpoint, 48. So a 60-month loan whose value wobbles slightly below 60 is routed as a 36-month loan. A dtype or rounding change in a production feed would cost `xgb_hist` 0.012 ROC-AUC. Encoding `term` as a categorical or binary field removes the risk.
4. **Ranking is robust to rate shocks, decisions are not.** A +3 pp rate rise leaves AUC within 0.0015 but moves approval by up to 8 pp. That supports monitoring approval rate and calibration-in-the-large, not AUC.

---

## 5. Analysis F — Interpretability (SHAP)

**Method** (`src/analysis/block4_shap.py`):
- **Sample:** a seeded 5,000-row Validation sample, explained in log-odds.
- **Boosters:** TreeSHAP for `xgb_hist`, `lgbm_native_flags`, and `lgbm_monotone`; CatBoost's native `ShapValues` on a `Pool` with its 5 categorical features.
- **Common vocabulary:** the 74–137 transformed inputs (one-hot levels and `missingindicator_*` flags) are summed back to the 50 raw fields plus 4 ratios, so every model is explained in the same terms.
- **Exact Champion attribution.** `stack_lr` is linear in member logits: $\operatorname{logit}(p_{\text{stack}}) = 0.2685 + 0.6514\,\ell_{\text{cat}} + 0.1811\,\ell_{\text{xgb}} + 0.2554\,\ell_{\text{lr}}$. Each member logit is exactly its SHAP sum (TreeSHAP for the boosters; $\beta_j (z_j - \bar z_j)$ for the logistic member). Therefore $\sum_k w_k \phi_k(\text{field})$ is an **exact** additive attribution of the Champion, with additivity error 1.6e-6.

**Global importance** (mean |SHAP|, log-odds; `artifacts/p4/tables/shap_importance.csv`):

| Field | `stack_lr` | `catboost` | `xgb_hist` | `lgbm_native_flags` | `lgbm_monotone` |
|---|---:|---:|---:|---:|---:|
| `term` | **0.286** (1) | **0.260** (1) | 0.254 (2) | 0.247 (2) | 0.224 (2) |
| `grade` | 0.214 (2) | 0.119 (4) | (10) | (8) | (9) |
| `acc_open_past_24mths` | 0.161 (3) | 0.154 (3) | 0.141 (3) | 0.134 (3) | 0.127 (3) |
| `int_rate` | 0.124 (4) | 0.198 (2) | **0.319** (1) | **0.294** (1) | **0.351** (1) |
| `home_ownership` | 0.115 (5) | 0.104 (5) | 0.102 (4) | 0.098 (4) | 0.095 (4) |
| `dti` | 0.108 (6) | 0.096 (6) | 0.095 (6) | 0.091 (5) | 0.089 (5) |

**Findings:**
1. **The models agree on what matters.** Spearman rank agreement of field importance is 0.92–0.99 across all 5 models (`artifacts/p4/tables/shap_rank_agreement.csv`). Loan terms, recent credit-seeking (`acc_open_past_24mths`), housing, and affordability dominate.
2. **The Champion splits the pricing signal differently.** The one-hot boosters concentrate it in `int_rate`, while `stack_lr` places `grade` 2nd and `sub_grade` 11th (vs 28th–34th for one-hot LightGBM/XGBoost), through CatBoost's target statistics and the logistic member. This matches Analysis A: grade and rate are interchangeable carriers of one signal.
3. **Monotone constraints show up in SHAP** (`artifacts/p4/tables/shap_monotone_check.csv`, `artifacts/p4/plots/shap_dependence_monotone.png`). All 7 constrained fields have the domain sign in both LightGBM variants, and the constraints tighten value-to-SHAP rank correlation most where the unconstrained model was noisiest: `revol_util` 0.819 → 0.925, `annual_inc` −0.884 → −0.948, `payment_to_income` 0.932 → 0.962. The unconstrained model's high-income outliers (above $150k with positive SHAP) disappear.
4. **A missing `emp_length` signals risk.** Borrowers with no employment length form a distinct cluster at about +0.4 log-odds in `artifacts/p4/plots/shap_beeswarm_stack_lr.png`, which suggests a fairness and data-quality review.
5. **Borderline decisions are explainable** (`artifacts/p4/tables/shap_local.csv`, `artifacts/p4/plots/shap_local_{approved,rejected}.png`). Two borrowers sit within 1e-5 of the Champion's 10:1 cutoff (0.1017). The approved borrower's 60-month term adds +0.61 log-odds, offset by a $250k income (−0.17), a $67.8k bankcard limit (−0.17), and few recent accounts (−0.16).
6. **Caveat on group totals.** `artifacts/p4/tables/shap_group_importance.csv` sums |SHAP| per group, which favours large groups (38 bureau fields). Analysis A's ablation is the size-neutral measure of group value.

---

## 6. Recommendations

| Use case | Recommendation | Evidence |
|---|---|---|
| Automated cost-based cutoffs | Deploy **`stack_lr` uncalibrated** with the Bayes rule $1/(1+R)$ or P3's tuned cutoff | Test ECE 0.0045; monthly ECE ≤ 0.0104; Platt adds nothing (§2) |
| Low-latency scoring (`xgb_hist` / `catboost`) | Deploy with the **Platt map fitted on the latest labelled vintage** | ECE −68% to −65%, significant Brier gain (§2) |
| Any model trained with class weights, undersampling, or AdaBoost | **Calibration is mandatory** before using cost cutoffs | −148 cost/1k at 5:1 (§2) |
| Production monitoring | Track **approval rate, calibration-in-the-large, `int_rate` PSI, and bureau-field missingness**, not only AUC | Rate shocks move approval up to 8 pp at \|ΔAUC\| ≤ 0.0015; NaN injection halves `xgb_hist` approvals (§4) |
| Model hygiene | Encode `term` as categorical; do not ship `logreg_baseline` with both `int_rate` and `sub_grade` | XGBoost `term < 60` split; logistic `int_rate` sign flip (§4b) |

---

## 7. Presentation slide outlines (Block 4 — 4 slides)

### Slide 1: Calibration — fixing probabilities so cost cutoffs work
- **Visuals:** `artifacts/p4/plots/reliability_test_catboost_imb_class_weights.png` + `artifacts/p4/plots/reliability_test_xgb_hist.png`
- **Bullets:**
  - Monotone calibration leaves ranking unchanged; its value is that the Bayes rule $t^* = 1/(1+R)$ becomes optimal.
  - Boosters: ECE 0.024–0.029 → 0.008–0.009; intercept +0.28/+0.34 → about 0 (the Train → Test default-rate shift).
  - Class-weighted CatBoost: Brier 0.2013 → 0.1387; cost −148 per 1,000 applicants at 5:1. The Champion needed no calibration (ECE 0.0045).
  - Selection rule fixed on cross-fitted Validation and locked by SHA-256 before the single Test read-out.

### Slide 2: Analysis A — what information the model really uses
- **Visual:** `artifacts/p4/plots/ablation.png`
- **Bullets:**
  - `full` arm reproduces P3's `xgb_hist` exactly (max |Δp| = 0).
  - Affordability −0.0214 ROC; bureau −0.0122; grade + `int_rate` −0.0066; grade alone −0.0003 (not significant).
  - LC pricing is a redundant carrier of risk; P2's ratios add nothing to trees.

### Slide 3: Analysis D — stability and stress tests
- **Visuals:** `artifacts/p4/plots/temporal_stability_test.png` + `artifacts/p4/plots/sensitivity.png`
- **Bullets:**
  - Monthly Test ROC-AUC 0.728–0.748, `logreg_baseline` last every month; boosters under-predict by 2–4 pp every month, the stack by under 1 pp.
  - Bureau-field "drift" is a pre-2012 reporting regime; the real drift is `int_rate` (PSI 0.179).
  - Missing data is the main risk (`xgb_hist` approvals 30% → 14%); a rate shock flips `logreg_baseline`'s decisions the wrong way.

### Slide 4: Analysis F — explaining the Champion exactly
- **Visuals:** `artifacts/p4/plots/shap_beeswarm_stack_lr.png` + `artifacts/p4/plots/shap_local_approved.png`
- **Bullets:**
  - Stack log-odds are linear in member logits, so the weighted member SHAP is an exact attribution (error 1.6e-6).
  - Top drivers: `term`, `grade` / `int_rate`, `acc_open_past_24mths`, `home_ownership`, `dti`; models agree (ρ = 0.92–0.99).
  - Monotone LightGBM tightens SHAP direction for `revol_util` (0.82 → 0.93) and `annual_inc` (−0.88 → −0.95).
