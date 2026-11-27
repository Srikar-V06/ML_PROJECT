# Memorandum — Cost-Sensitive Credit Underwriting Policy

**To:** Head of Credit Risk  
**From:** Credit Analytics & ML Team (Block 5 — Decision Policy & Risk Controls)  
**Date:** 2026-11-27  
**Subject:** Recommended Approval Threshold & Governance Controls for `stack_lr` Underwriting Model  

---

## 1. Executive Recommendation

Deploy the calibrated **`stack_lr` ensemble** (`Test ROC-AUC = 0.7439`, `PR-AUC = 0.4279`, `ECE = 0.0045`) using the **data-driven Bayes profit-maximizing rule** (`P(default) < 0.2155`, corresponding to a **3.64:1** LGD-to-net-margin cost ratio) when maximizing risk-adjusted portfolio return, or **`P(default) < 0.0909`** when operating under a conservative **10:1** capital-preservation mandate.

On the held-out `45,000`-borrower Test cohort (`2015-09` to `2015-12`, `19.91%` base default rate):
- **Status quo (`approve_all`, LC's historical book):** `100%` approval, `19.91%` default rate, **\$4.68** net profit per applicant (**\$210k** total).
- **Recommended data-driven Bayes policy (`p < 0.2155`):** `64.9%` approval, `11.37%` default rate among approved, **\$641.61** net profit per applicant (**\$28.87M** total; **+\$636.93/applicant** over status quo, and **+\$146.88/applicant** over a simple `Grades A–C` rule).
- **Conservative 10:1 capital-preservation policy (`p < 0.0909`):** `26.2%` approval, `5.22%` default rate among approved, reducing 10:1 unit cost from `1,990.9` to **`689.0` per 1,000 applicants** (`93.1%` default recall) while still generating **\$440.33/applicant** net profit.

## 2. Cost Assumptions & Sensitivity

Using `210,000` historical Training loans (`artifacts/p5/tables/cost_data_driven.csv`):
- **Loss Given Default (LGD):** `55.80%` mean (`(funded_amnt - total_rec_prncp - recoveries) / funded_amnt` on charged-off loans).
- **Net Interest Margin:** `15.33%` mean (`19.33%` gross interest collected on fully paid loans minus `4.00%` assumed funding cost over average duration).
- **Implied Cost Ratio:** `0.5580 / 0.1533 = 3.64 : 1` (theoretical Bayes cutoff $t^* = 1/(1+R) = 0.2155$).

**Sensitivity (`2:1` to `20:1`, `artifacts/p5/tables/cost_sensitivity.csv`):** Because `stack_lr` is already well-calibrated (`ECE = 0.0045`), the theoretical Bayes rule $t^* = 1/(1+R)$ tracks the Validation-tuned empirical optimum closely across the entire `2:1–20:1` range. As $R$ increases from `2:1` to `20:1`, optimal approval drops from `87.1%` (`t=0.333`) to `9.3%` (`t=0.048`). For standalone boosters (`catboost`, `xgb_hist`, `lgbm_native_flags`), **Platt calibration is mandatory** before applying $1/(1+R)$.

## 3. Key Risks, Limitations & Fairness

1. **Reject Inference Bias:** All training and test loans were *already approved* by Lending Club. Model rankings are validated on LC-eligible applicants; performance on previously rejected populations is unobserved.
2. **Irreducible Default Ceiling:** Error analysis shows `484` of `8,959` Test defaults (`5.4%`) are approved by *all five* tested models (`mean FICO = 711.0`, `mean DTI = 15.9%`, `mean income = \$105k`), reflecting post-origination shocks (job loss, medical events) outside application data.
3. **Fairness & ECOA Compliance:** `zip_code` is excluded from all models. Regional approval rates (`26.0%–27.0%`) and equalized-odds gaps (`1.46 pp`) are balanced. Income-band approval disparities (`11.9%` in Q1 vs `41.1%` in Q4) reflect underlying default-rate differences (`22.6%` vs `16.4%`); `fairlearn.ThresholdOptimizer` can cut the income equalized-odds gap from `32.3 pp` to `3.8 pp`, but raises 10:1 cost from `689` to `1,042` per 1k.
4. **Cross-Lender Domain Shift:** Zero-shot transfer to Prosper (`44,320` loans, `32.7%` default rate) drops ROC-AUC from `0.736` to `0.657–0.674` and requires target-platform Platt recalibration (`ECE 0.109 → 0.040`).

## 4. Monitoring & Governance Plan

- **Quarterly Calibration & PSI Audit:** Re-check `stack_lr` ECE and `int_rate` PSI (`0.179` on Test) quarterly; re-fit Platt scaling if `ECE > 0.02` or base default rate shifts `> 2 pp`.
- **Latency & Fallback:** Deploy `stack_lr` (`42.1 ms/1k` single-thread) as primary, with `catboost + Platt` (`19.4 ms/1k`) or `xgb_hist + Platt` (`12.6 ms/1k`) as low-latency fallback.
