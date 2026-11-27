# Block 5 Report Sections & Slide Outlines (Person 5 — Business ML, Fairness, Cross-Lender & UI)

Block 5 completes the end-to-end decisioning, error/prediction diagnostics, fairness governance, cross-lender generalization, and production UI/API layers without modifying any frozen Block 3 or Block 4 artifacts (`artifacts/p3/test_lock.json`, `artifacts/p4/calibration_lock.json`). All Block 5 analyses are reproducible via `python -m src.block5 --stage all`.

---

## 1. Cost-Sensitive Threshold Study (`src/analysis/block5_cost.py`)

Using `210,000` Training loans joined to `audit.parquet` (`artifacts/p5/tables/cost_data_driven.csv`):
- **Empirical LGD (charged-off Train loans):** `55.80%` mean (`59.07%` median).
- **Empirical Interest Margin (fully-paid Train loans):** `19.33%` gross, **`15.33%` net** after `4.00%` assumed funding cost.
- **Data-driven cost ratio:** `3.64 : 1` (net), yielding Bayes threshold $t^* = 1/(1+R) = 0.2155$ (vs `0.0909` at `10:1`).

| Test Policy (`n=45,000`) | Approval Rate | Default Rate (Approved) | 10:1 Cost / 1k | Net Profit / Applicant |
|---|---:|---:|---:|---:|
| `approve_all` (status quo) | 100.0% | 19.91% | 1,990.9 | \$4.68 |
| `grades_A_to_C_only` | 78.6% | 14.81% | 1,295.8 | \$494.73 |
| `stack_lr\|naive_050` | 95.6% | 18.08% | 1,745.9 | \$236.45 |
| `stack_lr\|bayes_10_to_1` (`p<0.0909`) | 26.2% | 5.22% | **689.0** | \$440.33 |
| `stack_lr\|bayes_data_driven` (`p<0.2155`) | 64.9% | 11.37% | 963.3 | **\$641.61** |
| `stack_lr\|fixed_approval_80` | 82.5% | 14.52% | 1,293.6 | \$555.19 |

Per-loan dollar expected-profit (`(1-p)*margin*loan_amnt > p*LGD*loan_amnt`) simplifies algebraically to `p < margin / (margin + LGD) = 0.2155`, matching `bayes_data_driven` identically. Across `2:1–20:1` (`artifacts/p5/tables/cost_sensitivity.csv`, `cost_sensitivity.png`), `stack_lr`'s theoretical Bayes cutoff tracks the Validation-tuned empirical cutoff closely because its `Test ECE = 0.0045`.

---

## 2. Analysis E (Error Analysis) & Analysis B (Prediction Visualization) (`src/analysis/block5_errors.py`)

- **FN vs TN/TP Profiles (`error_profiles.csv`):** At `10:1` (`t=0.0909`), `stack_lr` produces `616` False Negatives (`1.37%` of Test) with `mean FICO = 711.0`, `DTI = 15.9%`, `int_rate = 8.08%`, `annual_inc = \$105.1k` — virtually indistinguishable at origination from True Negatives (`FICO = 718.4`, `DTI = 15.6%`, `int_rate = 7.64%`), whereas True Positives average `FICO = 684.3`, `DTI = 21.5%`, `int_rate = 15.02%`.
- **Cross-Model FN Consensus (`error_fn_overlap.csv`, `error_fn_overlap.png`):** Of `8,959` Test defaults, **`484`** are missed by **all 5** models (`stack_lr`, `catboost_platt`, `xgb_hist_platt`, `lgbm_native_flags_platt`, `logreg_baseline`), whereas only `159` are unique to 1 model. Pairwise FN Jaccard overlap among tree/stack models is `0.73–0.92`, confirming an irreducible data ceiling from post-origination shocks.
- **Confidence & Segments (`error_confidence.png`, `error_segments.png`):** Decision error rates peak near the `0.0909` threshold; `Grade A` has the highest FNR (low predicted risk) while `Grade E–G` has the highest FPR.
- **Analysis B (`pred_histograms.png`, `gain_lift.png`, `pred_vs_actual_grade_vintage.png`, `confusion_matrices.png`):** `stack_lr` and Platt-calibrated boosters align closely with actual default rates across Grades `A–G` and all Test monthly vintages.

---

## 3. Fairness Audit & Fairlearn Mitigation (`src/analysis/block5_fairness.py`)

Lending Club data contains no race or sex attributes, and `zip_code` was excluded per ECOA. Auditing `stack_lr` at `10:1` across 4 available proxy/demographic-adjacent dimensions (`fairness_audit.csv`, `fairness_groups.png`):
- **US Census Region:** Balanced — approval `26.0%–27.0%`, TPR `92.9%–93.5%`, DP gap `0.95 pp`, max EO gap `1.46 pp`.
- **Employment Length Missingness:** Missing `emp_length` (`n=2,923`) has `25.25%` actual default rate vs `19.54%` present; approval is `13.2%` vs `27.2%`.
- **Income Band (`Q1_low`–`Q4_high`) & Home Ownership:** `Q1_low` (`22.6%` default rate) has `11.9%` approval vs `Q4_high` (`16.4%` default rate) `41.1%` approval (`DP gap = 29.2 pp`, `EO gap = 32.3 pp`).
- **Mitigation (`fairlearn.ThresholdOptimizer`, `fairness_mitigation.csv`):** Fitted on Validation and evaluated on Test, `equalized_odds` cuts the income-band EO gap from **`32.3 pp → 3.8 pp`** (DP gap `29.2 pp → 1.5 pp`), while raising 10:1 cost from `689.0` to `1,042.2` per 1k (because `ThresholdOptimizer` maximizes balanced accuracy rather than 10:1 asymmetric cost).

---

## 4. Cross-Lender Generalization: Lending Club → Prosper (`src/analysis/block5_cross_lender.py`)

Models trained on `210,000` LC Train rows using the `16` aligned common features (`common_16_no_grade`) were evaluated on `35,456` held-out Prosper Test loans (`32.7%` overall Prosper default rate, chronological 20%/80% calib/test split):

| Model | LC Test ROC | Prosper Zero-Shot ROC | Prosper Zero-Shot ECE (Raw → Platt) | Prosper Cost/1k (Raw → Platt) |
|---|---:|---:|---:|---:|
| `logreg` | 0.7317 | **0.6738** | 0.0717 → **0.0426** | 724.0 → 692.3 |
| `xgb` | 0.7361 | 0.6702 | 0.0930 → **0.0561** | 704.9 → 675.3 |
| `lgbm` | 0.7358 | 0.6634 | 0.1027 → **0.0580** | 703.5 → **654.2** |
| `catboost` | **0.7364** | 0.6567 | 0.1087 → **0.0395** | 748.2 → 657.8 |

Key findings: (1) Including mapped platform grades (`ProsperRating` → `LC grade`) **hurts** zero-shot ROC (`0.6634 → 0.6485` for `lgbm`); (2) Target-platform Platt scaling on `8,864` Prosper calibration rows cuts ECE by `41–64%` and reduces 10:1 cost by `30–90` per 1k; (3) Refitting `lgbm` on `8,864` Prosper rows recovers ROC-AUC to `0.6815` (`PR-AUC 0.4685`).

---

## 5. Production UI, FastAPI Service & Containerization

- **5-Page Streamlit App (`app/app.py`, `app/scoring.py`):** Single applicant scoring with live SHAP attribution & What-If curve; Batch CSV scoring with schema validation; Model Benchmark Dashboard; Cost-Sensitive Policy Explorer; Insights/Fairness/Cross-Lender tabs.
- **FastAPI `/predict` & `/health` (`app/api.py`) + `Dockerfile`:** Containerized REST + UI deployment; verified by `tests/test_app_smoke.py` and `tests/test_block5.py`.

---

## Slide Outlines (3 Slides for Block 5)

1. **Slide 1 — Cost-Sensitive Underwriting & Policy Impact:** Data-driven LGD (`55.8%`) & net margin (`15.3%`) → `3.64:1` ratio (`t*=0.2155`); `stack_lr` lifts Test net profit from **\$4.68** (`approve_all`) to **\$641.61/applicant** (`+\$28.7M` on 45k loans); sensitivity across `2:1–20:1`.
2. **Slide 2 — Error Analysis & Fairness Governance:** `484` hard FNs missed by all 5 models (`mean FICO 711`, post-origination shocks); regional parity (`1.46 pp` EO gap) vs income-band EO gap (`32.3 pp → 3.8 pp` via `fairlearn`, cost trade-off).
3. **Slide 3 — Cross-Lender Transfer & Live UI/API:** LC → Prosper (`0.736 → 0.657–0.674` ROC), why platform grades hurt transfer, Platt recalibration (`ECE −64%`), and 5-page Streamlit + FastAPI architecture.
