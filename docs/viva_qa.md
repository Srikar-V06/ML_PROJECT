# Viva | Block 1

1. **Why exclude Current?** Its terminal outcome is unknown. Calling it good would reward vintages with insufficient follow-up.
2. **Why 2016Q1?** It is the last quarter of the contiguous window passing the 90% 36-month resolution rule. The cutoff quarter achieved 93.41%.
3. **What remains censored?** Only 58.65% of 60-month loans at the cutoff were resolved. Early exits can have different risks from loans still active.
4. **What does 200k represent?** It is a proportional sample of 938,540 eligible accepted and resolved loans. Rejected applicants have no observed outcome here.
5. **How close was the sample?** Maximum PSI was 0.0000643 and maximum annual rate gap 0.001029. PSI uses binned distributions; continuous KS uses the empirical distributions.
6. **Can the same person appear twice?** Possibly. The tape has no observed member identifiers. Unique loan IDs guarantee distinct loans.
7. **How do you catch leakage that looks predictive?** Inventory timing first, screen one feature at a time on Train and then deliberately contaminate a diagnostic fit. Its validation AUC was 0.99964.
8. **Why keep almost-empty variables?** Coverage can encode timing, product or borrower context. 29 were empty throughout Train; P2 can inspect later coverage and choose appropriate fold-local handling.
9. **Does missingness prove MNAR?** The two-proportion test measures an association in observed defaults. It cannot establish the missing-data mechanism or rule out vintage confounding.
10. **Why both OOT and expanding CV?** OOT tests transfer to later vintages. The four Train-only folds permit repeated model development without accessing final assessment rows.
11. **How does one scorecard point arise?** Each bin contributes −(20/ln 2) × coefficient × WoE. Add the intercept-adjusted base. Doubling good/bad odds adds 20 points.
12. **How did you prevent target leakage in WoE?** Fit boundaries, WoE, IV selection and logistic coefficients on Train. A value unseen in later data uses its learned bin or a neutral fallback.
13. **Why paired DeLong?** The same loans receive both scores. Paired placements retain covariance. Scorecard versus full logistic produced p=1.95e-07; paired bootstrap also compares AP and Brier.
14. **What does a 10:1 cost ratio buy?** It values one false negative at ten false positives. The locked threshold 0.09684 cost 697.0 relative units per 1,000 later loans. Real exposure and foregone margin determine actual dollars.
15. **Can P2 apply the scorecard directly to Prosper?** Only 17 underwriting fields have mapped counterparts. A transfer experiment requires harmonized categories and units, a common-feature model and assessment of label and vintage shift.

## Block 2

1. **Why multiply installment by twelve?** Installment is a monthly payment and income is annual. The ratio expresses the share of reported income committed to this loan. A zero income leaves that ratio undefined.
2. **What does a tree split optimize?** The default Gini criterion favors a split whose children have purer outcome distributions. Entropy is another measure of class uncertainty. Minimum leaf size limits how small a segment can become, and cost-complexity pruning penalizes extra leaves.
3. **Why use out-of-bag estimates and expanding folds?** Each bootstrap tree omits some training loans, so those trees can score their omitted loans. This measures internal forest performance. Expanding folds answer a different question about performance on subsequently issued loans.
4. **Why can two importance rankings disagree?** Impurity importance rewards features that yield many useful training splits. Permutation importance measures validation AUC lost when one input is shuffled. A correlated substitute can keep the AUC high.
5. **Why might 95% explained variance still lose predictive information?** PCA learns orthogonal directions that capture variation in inputs. A low-variance direction can carry default signal, and rotation can turn a single underwriting cutoff into several tree splits. Compare each PCA model against its matched control on Val.
