# Block 1 | Nine-slide outline

| Slide | Evidence and speaking point | Visual under artifacts/plots |
| --- | --- | --- |
| 1. What the decision covers | 2,260,668 accepted loans; outcomes are terminal repayment or default. Loan IDs are unique and member IDs unavailable. | censoring_by_vintage.png |
| 2. Choose the mature window | At 2016Q1, 36-month resolution was 93.4% and 60-month resolution was 58.6%. | censoring_by_vintage.png |
| 3. Sample the eligible pool | Draw 200,000 from 938,540; max PSI 0.000064. | sampling_representativeness.png |
| 4. Show leakage inflation | 99 approved fields; compare clean and contaminated validation AUC (0.737 and 1.000). | leakage_comparison.png |
| 5. Retain missingness | 41 Train-derived indicators; employment history illustrates selection. | missingness_default_rates.png |
| 6. Freeze time and compare | 140k / 30k / 30k; full logistic Test AUC 0.742, AP 0.417. | comparison_roc.png, comparison_pr.png |
| 7. Translate odds into points | 30 IV-selected fields; 600 points at 10:1 good odds, 20 points per doubling. | scorecard_distribution.png |
| 8. Choose a threshold | At 10:1, 0.0968 gives 30.0% approval and 697.0 cost per 1,000. | comparison_cost.png, comparison_calibration.png |
| 9. P2 and external transfer | Four expanding training folds, fixed paired prediction contract and 17 Prosper mappings for 55,071 terminal loans. | comparison_distribution.png |
