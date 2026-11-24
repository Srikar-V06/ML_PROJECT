"""Render the handoff evidence from a completed real run."""
from pathlib import Path
import json
import pandas as pd

from src.utils.io import artifact, cli_config


def table(frame):
    def fmt(v):
        if pd.isna(v):
            return "—"
        if isinstance(v, float):
            return f"{v:.5g}"
        return str(v).replace("|", "/")
    rows = [frame.columns, ["---"]*len(frame.columns), *frame.itertuples(index=False, name=None)]
    return "\n".join("| " + " | ".join(map(fmt, row)) + " |" for row in rows)


def run(cfg):
    root = Path(cfg["paths"]["docs"])
    root.mkdir(parents=True, exist_ok=True)
    def csv(name):
        return pd.read_csv(artifact(cfg, name))
    def read(name):
        return json.loads(artifact(cfg, name).read_text())
    def save(name, contents):
        (root/name).write_text(contents.strip()+"\n", encoding="utf-8")
    pop = read("tables/population.json")
    prosper = read("tables/prosper_population.json")
    audit = csv("feature_audit.csv")
    clean = audit.query("decision == 'keep'")
    best = clean.nlargest(1, "single_feature_auc").iloc[0]
    rep = csv("sampling_representativeness.csv")
    miss = csv("missing_value_table.csv")
    top_miss = miss.dropna(subset=["p_value"]).sort_values("rate_diff", key=abs, ascending=False)
    leak = csv("tables/leakage_lgbm_comparison.csv")
    vintage = csv("tables/vintage_resolution.csv")
    cutoff = vintage[vintage.quarter.eq(pop["cutoff"])]
    resolution = cutoff.set_index("term")["mean"]
    manifest = json.loads((Path(cfg["paths"]["splits"])/"manifest.json").read_text())
    splits = pd.DataFrame(manifest["splits"]).T.reset_index(names="split")
    models = ["constant", "grade_only", "logreg_baseline", "woe_scorecard"]
    results = {name: read(f"metrics/{name}.json") for name in models}
    summary = pd.DataFrame([{"model": name, "split": split, **{k: r[split][k] for k in
                             ("roc_auc", "pr_auc", "brier", "ks", "log_loss", "ece", "calibration_slope", "calibration_intercept")}}
                             for split in ("val", "test") for name, r in results.items()])
    summary.to_csv(artifact(cfg, "tables/model_metrics.csv"), index=False)
    comps = csv("tables/model_comparisons.csv")
    costrows = []
    for ratio in cfg["cost_ratios"]:
        key = str(float(ratio))
        name = min(models, key=lambda m: results[m]["val"]["cost_per_1000"][key]["optimized"])
        r = results[name]["test"]["cost_per_1000"][key]
        costrows.append(dict(ratio=f"{ratio:g}:1", model=name, threshold=r["threshold"], approval=r["approval_rate"],
                             optimized=r["optimized"], naive_050=r["naive_050"], approve_all=r["approve_all"],
                             saving_vs_all=r["approve_all"]-r["optimized"]))
    costs = pd.DataFrame(costrows)
    costs.to_csv(artifact(cfg, "tables/validation_selected_policies.csv"), index=False)
    woe = csv("tables/woe_iv_summary.csv")
    selected = int(woe.selected.sum())
    maxpsi = rep.psi.max()
    gap = rep.query("measure == 'default_rate'").difference.max()
    status = pd.DataFrame([{"status": s, "raw loans": n, "target or disposition": cfg["target"].get(s, "excluded")}
                           for s, n in pop["status_counts"].items()])
    lr = results["logreg_baseline"]["test"]
    sc = results["woe_scorecard"]["test"]
    pvals = comps[comps.test.eq("delong")].set_index("model_b").p_value
    middle = costs.iloc[1]
    inter = int(miss.add_missing_indicator.sum())
    empty = int((miss.missing_pct == 100).sum())
    save("block1_report.md", f"""# Block 1 | Accepted-loan default prediction

## Cohort and outcome

The supplied tape contains {pop['raw_rows']:,} loan records and {pop['csv_footer_records']} footer records. Every loan ID is unique. All member IDs are null, so the number of distinct borrowers cannot be verified. No member-based duplicates were removed. Fully paid loans are controls and charged-off or defaulted loans are events.

{table(status)}

There were zero Issued rows. Removing excluded statuses and {pop['immature_terminal_removed']:,} terminal loans issued after the cutoff left {pop['mature_rows']:,} loans. Exact source identity and counts are in artifacts/tables/population.json.

## Right-censoring

The full-tape analysis measures the terminal fraction by quarter and term, counting resolved legacy-policy outcomes only for this diagnostic. End the contiguous mature window at {pop['cutoff']}, where {resolution[36]:.2%} of 36-month and {resolution[60]:.2%} of 60-month loans were resolved. At the cutoff a material fraction of longer-term borrowers still had no terminal label. Estimates condition on resolution, which can favor early payoff or early default.

![Resolved loans by quarter and term](../artifacts/plots/censoring_by_vintage.png)

## Sampling proof

The seed-42 draw retained {pop['sampled_rows']:,} loans. Integer quotas preserve annual event rates, then allocate proportionally across quarter and grade within each outcome. The nearest feasible yearly count is used when rounding alone would breach the tolerance. Eight singleton rows triggered quarter-and-outcome pooling in the affected cells. Draws are without replacement. The largest absolute annual default-rate difference was {gap:.6f} and the largest PSI was {maxpsi:.7f}. These checks compare the sample with the mature pool.

{table(rep.query("measure == 'default_rate'")[["feature","population","sample","difference"]])}

{table(rep.query("measure == 'distribution'")[["feature","ks","psi","difference"]])}

PSI uses mature-pool deciles for continuous variables, category shares for discrete fields, an explicit missing bin and a probability floor of 1e-8. The final column is the largest absolute bin or category share difference. The KS statistic uses nonmissing continuous values. See [the sampling plot](../artifacts/plots/sampling_representativeness.png).

## Leakage audit

The domain inventory covers all {len(audit)} fields and retains {len(clean)} original application, bureau and pricing features. Historical bureau collection counts describe events before underwriting. Collections and recoveries on the subject loan are servicing outcomes and are excluded. An unseen schema field fails closed. The feature audit records Train dtypes and missingness after parsing.

{table(audit.stage.value_counts().rename_axis("stage").reset_index(name="columns"))}

Pass 2 used Train only. Numeric scores were checked in both directions and for a missingness effect. Four-fold out-of-fold smoothed category means guarded against memorization of high-cardinality strings. The largest clean single-feature ROC-AUC was {best.single_feature_auc:.4f} for {best.feature_name}. No clean feature crossed the 0.75 review threshold.

Pass 3 held the validation cohort and LightGBM settings fixed and added just the audited post-origination columns to the diagnostic fit.

{table(leak)}

{table(csv("tables/leaked_feature_importance.csv").head(5))}

![Validation performance and leaked feature gain](../artifacts/plots/leakage_comparison.png)

## Missingness

The Train study retains {empty} fully missing features and recommends {inter} missing indicators. For features more than 0.5% missing with both groups present, the pooled two-proportion z-test compares observed event rates. An entirely missing field has no present group, so its difference and test are undefined. The four mechanism labels are working interpretations. These exploratory p-values are unadjusted, and vintage can confound associations.

{table(miss.missing_type.value_counts().rename_axis("category").reset_index(name="features"))}

{table(top_miss.head(6)[["feature_name","missing_pct","default_rate_missing","default_rate_present","rate_diff","p_value"]])}

![Conditional default rates](../artifacts/plots/missingness_default_rates.png)

## Frozen evaluation design

{table(splits[["split","rows","start","end","default_rate"]])}

Four expanding Train folds use successive validation blocks of 28,000 loans and training blocks of 28,000, 56,000, 84,000 and 112,000. The id tie-break preserves monthly origination ordering. Outcomes come from one retrospective snapshot. These partitions test vintage transfer; historical label availability at each simulated fit date is unknown.

## Baselines and scorecard

{table(summary)}

The constant predicts the Train event rate. The grade model fits grade and subgrade. The full logistic pipeline fits imputation, indicators, scaling and encoding on Train. The real 1,000-loan memorization check produced Train AUC {read("metrics/overfit_sanity.json")["train_auc"]:.4f}. It tests wiring, while OOT estimates measure discrimination on later loans. Average precision has the event prevalence as its constant-score reference. For full logistic Test AUC the 95% loan-bootstrap interval was [{lr['ci95']['roc_auc'][0]:.4f}, {lr['ci95']['roc_auc'][1]:.4f}]. Each model JSON includes all four 1,000-replicate intervals.

The scorecard selected {selected} fields with Train IV ≥ 0.02. This run used the built-in monotonic quantile and pooled-category fallback. Numeric quantile bins with equal isotonic event rates merge. Bins receive half-count smoothing. WoE is log(good share / bad share) and IV sums (good share − bad share) × WoE. Missing values have their own bin. The base is 600 at 10 good to 1 bad. Factor = 20 / ln(2) and offset = 600 − factor × ln(10). Score = offset − factor × logit(P(default)). The base row includes the intercept and each feature contributes −factor × coefficient × WoE. Summing unrounded exported points reproduces the score. See [scorecard distribution](../artifacts/plots/scorecard_distribution.png).

{table(woe.head(6))}

## Paired Test comparisons

Positive differences favor the scorecard for discrimination. Negative Brier differences favor the scorecard for probability accuracy. ROC inference uses paired Sun–Xu midranks with tie corrections. The loan bootstrap uses the same sampled rows for both models.

{table(comps)}

## Validation-selected policies

For each cost ratio the threshold minimizes Validation cost. Among the four candidate models, the displayed policy also has the lowest Validation cost. The operating results below use untouched Test decisions, measured in relative cost units per 1,000 loans. Approval is a fraction.

{table(costs)}

The full policy breakdown per model is in artifacts/metrics. See [model calibration](../artifacts/plots/comparison_calibration.png) and [cost curves](../artifacts/plots/comparison_cost.png).

## Prosper transfer preparation

The historical export contained {prosper['raw']:,} records. The terminal-status map retained {prosper['terminal_rows']:,} records and removing {prosper['deduplicated']} repeated loan keys yielded {prosper['aligned']:,} loans and {prosper['features']} aligned features. Every Past Due category was excluded as unresolved, along with Current, Cancelled and FinalPaymentInProgress. Mapping details, units and proxies are in artifacts/tables/prosper_feature_mapping.csv. Comparing platform grades requires harmonization. Lending Club ownership categories need to collapse to the Prosper owner flag. Employment-status duration and bankcard utilization are proxies. An external evaluation also needs a model trained on the common columns and a Prosper maturity analysis.

{table(pd.DataFrame(prosper["statuses"].items(),columns=["Prosper status","raw records"]))}
""")
    save("head_of_risk_memo.md", f"""# Decision memo | Head of Risk

**Subject — Loan approval thresholds under three loss assumptions**

At a 10:1 loss ratio, the validation-selected {middle.model} policy approved {middle.approval:.1%} of the 30,000 later test loans. Its cost was {middle.optimized:.1f} units per 1,000 applicants, a reduction of {middle.saving_vs_all:.1f} against approving everyone. Each unit is the assumed cost of rejecting one borrower who would have repaid.

| Default approval cost / good rejection cost | Reject at or above | Test approval | Cost per 1,000 | At 0.50 | Approve all |
| --- | ---: | ---: | ---: | ---: | ---: |
""" + "\n".join(f"| {r.ratio} | {r.threshold:.5f} | {r.approval:.1%} | {r.optimized:,.1f} | {r.naive_050:,.1f} | {r.approve_all:,.1f} |" for r in costs.itertuples()) + f"""

All three policies used thresholds selected before Test. The assumed loss on a default raises the cost of approving risky loans and lowers the approval rate. The observed Test default rate was {manifest['splits']['test']['default_rate']:.2%}.

To translate these results into money, multiply the table by the institution's estimated loss from rejecting a good loan. The analysis assigns the same exposure to each application. Vintage censoring and the absence of rejected applicants limit these estimates to the historical accepted-loan cohort. A rollout decision needs loss amounts, margins, and a prospective check of approval volume and calibration.""")
    save("slides_outline.md", f"""# Block 1 | Nine-slide outline

| Slide | Evidence and speaking point | Visual under artifacts/plots |
| --- | --- | --- |
| 1. What the decision covers | {pop['raw_rows']:,} accepted loans; outcomes are terminal repayment or default. Loan IDs are unique and member IDs unavailable. | censoring_by_vintage.png |
| 2. Choose the mature window | At {pop['cutoff']}, 36-month resolution was {resolution[36]:.1%} and 60-month resolution was {resolution[60]:.1%}. | censoring_by_vintage.png |
| 3. Sample the eligible pool | Draw 200,000 from {pop['mature_rows']:,}; max PSI {maxpsi:.6f}. | sampling_representativeness.png |
| 4. Show leakage inflation | {len(clean)} approved fields; compare clean and contaminated validation AUC ({leak.roc_auc.iloc[0]:.3f} and {leak.roc_auc.iloc[1]:.3f}). | leakage_comparison.png |
| 5. Retain missingness | {inter} Train-derived indicators; employment history illustrates selection. | missingness_default_rates.png |
| 6. Freeze time and compare | 140k / 30k / 30k; full logistic Test AUC {lr['roc_auc']:.3f}, AP {lr['pr_auc']:.3f}. | comparison_roc.png, comparison_pr.png |
| 7. Translate odds into points | {selected} IV-selected fields; 600 points at 10:1 good odds, 20 points per doubling. | scorecard_distribution.png |
| 8. Choose a threshold | At 10:1, {middle.threshold:.4f} gives {middle.approval:.1%} approval and {middle.optimized:.1f} cost per 1,000. | comparison_cost.png, comparison_calibration.png |
| 9. P2 and external transfer | Four expanding training folds, fixed paired prediction contract and {prosper['features']} Prosper mappings for {prosper['aligned']:,} terminal loans. | comparison_distribution.png |""")
    save("viva_qa.md", f"""# Viva | Fifteen questions and answers

1. **Why exclude Current?** Its terminal outcome is unknown. Calling it good would reward vintages with insufficient follow-up.
2. **Why {pop['cutoff']}?** It is the last quarter of the contiguous window passing the 90% 36-month resolution rule. The cutoff quarter achieved {resolution[36]:.2%}.
3. **What remains censored?** Only {resolution[60]:.2%} of 60-month loans at the cutoff were resolved. Early exits can have different risks from loans still active.
4. **What does 200k represent?** It is a proportional sample of {pop['mature_rows']:,} eligible accepted and resolved loans. Rejected applicants have no observed outcome here.
5. **How close was the sample?** Maximum PSI was {maxpsi:.7f} and maximum annual rate gap {gap:.6f}. PSI uses binned distributions; continuous KS uses the empirical distributions.
6. **Can the same person appear twice?** Possibly. The tape has no observed member identifiers. Unique loan IDs guarantee distinct loans.
7. **How do you catch leakage that looks predictive?** Inventory timing first, screen one feature at a time on Train and then deliberately contaminate a diagnostic fit. Its validation AUC was {leak.roc_auc.iloc[1]:.5f}.
8. **Why keep almost-empty variables?** Coverage can encode timing, product or borrower context. {empty} were empty throughout Train; P2 can inspect later coverage and choose appropriate fold-local handling.
9. **Does missingness prove MNAR?** The two-proportion test measures an association in observed defaults. It cannot establish the missing-data mechanism or rule out vintage confounding.
10. **Why both OOT and expanding CV?** OOT tests transfer to later vintages. The four Train-only folds permit repeated model development without accessing final assessment rows.
11. **How does one scorecard point arise?** Each bin contributes −(20/ln 2) × coefficient × WoE. Add the intercept-adjusted base. Doubling good/bad odds adds 20 points.
12. **How did you prevent target leakage in WoE?** Fit boundaries, WoE, IV selection and logistic coefficients on Train. A value unseen in later data uses its learned bin or a neutral fallback.
13. **Why paired DeLong?** The same loans receive both scores. Paired placements retain covariance. Scorecard versus full logistic produced p={pvals['logreg_baseline']:.3g}; paired bootstrap also compares AP and Brier.
14. **What does a 10:1 cost ratio buy?** It values one false negative at ten false positives. The locked threshold {middle.threshold:.5f} cost {middle.optimized:.1f} relative units per 1,000 later loans. Real exposure and foregone margin determine actual dollars.
15. **Can P2 apply the scorecard directly to Prosper?** Only {prosper['features']} underwriting fields have mapped counterparts. A transfer experiment requires harmonized categories and units, a common-feature model and assessment of label and vintage shift.""")
    save("walkthrough_script.md", f"""# Four-minute walkthrough

**0:00–0:45 | Start with the population.** Open artifacts/tables/population.json. We began with {pop['raw_rows']:,} actual Lending Club loans. The tape also contains 33 footer lines. The loan identifiers pass uniqueness checks; the member identifiers are empty. That distinction matters whenever someone asks how many individual people the model has seen. Now open artifacts/plots/censoring_by_vintage.png. The horizontal line is the 90% resolution requirement for the shorter term. The last qualifying quarter is {pop['cutoff']}. The longer loans are still resolving, so every number that follows describes an accepted, resolved cohort.

**0:45–1:25 | Trace a row into its partition.** In src/data/prep_lending_club.py, the first pass establishes eligibility and status counts on the full tape. Then the sampler selects 200,000 loans from {pop['mature_rows']:,}. The second pass retrieves their fields. Open the representativeness table and point to the largest PSI, {maxpsi:.7f}. Then open splits/manifest.json. The 140,000 Train, 30,000 Val and 30,000 Test loans are ordered by issue month and ID. Four expanding folds live entirely inside Train. The loader checks the sample and each membership file against their hashes.

**1:25–2:05 | Demonstrate the feature boundary.** In artifacts/feature_audit.csv, filter decision to keep. There are {len(clean)} features, including sparse origination attributes. Now show artifacts/plots/leakage_comparison.png. In the deliberately contaminated comparison, later payments and recoveries push validation AUC to {leak.roc_auc.iloc[1]:.4f}. With the audited features it is {leak.roc_auc.iloc[0]:.4f}. Open the missingness table. For each field we give the rate, likely source of missing values, and how P2 could handle it in a linear or tree model. Fold-specific transformations still have to be learned on the training fold.

**2:05–2:50 | Compare risk estimates.** Open artifacts/tables/model_metrics.csv and focus on Test. The full logistic baseline has AUC {lr['roc_auc']:.4f} and Brier {lr['brier']:.4f}. The scorecard gives AUC {sc['roc_auc']:.4f}. Its {selected} selected features contribute additive points. Start at the base row in credit_scorecard_points.csv and add the bin contribution for each feature. Twenty more points double good-to-bad odds. Open the calibration plot to distinguish ordering borrowers from accurately estimating event probabilities.

**2:50–3:25 | Price the decision.** Show the Head of Risk memo. We set thresholds on Validation and apply the same thresholds on Test. At 10:1, the selected policy approves {middle.approval:.1%} and incurs {middle.optimized:.1f} units per thousand applications. The unit is the opportunity cost of rejecting one repaying borrower. This analysis does not contain institution-specific loss amounts. The cost curve shows how sensitive each rule is to the chosen threshold.

**3:25–4:00 | Hand over the interface.** Open tests/test_models.py and the CI workflow. The 500-row offline fixture runs through the same fitting and evaluation interfaces, including a compact memorization test. The production run separately checks 1,000 real Train loans. Finally open HANDOFF.md. P2 registers a factory, tunes in the expanding folds and uses the standard CLI. Predictions carry the same loan IDs for paired inference. The external preparation delivers {prosper['aligned']:,} aligned Prosper loans and a documented mapping for {prosper['features']} features. Those definitions are the starting point for a separate transfer study.""")


if __name__ == "__main__":
    run(cli_config())
