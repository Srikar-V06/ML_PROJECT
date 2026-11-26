import json

import pandas as pd

from src.evaluation.stats_tests import compare_models
from src.evaluation.plots import plots
from src.utils.io import artifact, cli_config


def run(cfg):
    models = ["constant", "grade_only", "logreg_baseline", "woe_scorecard"]
    split = cfg.get("evaluation_splits", ["val", "test"])[-1]
    summary = []
    for name in models:
        metrics = json.loads(artifact(cfg, f"metrics/{name}.json").read_text(encoding="utf-8"))
        result = metrics[split]
        low, high = result["ci95"]["roc_auc"]
        summary.append(dict(model=name, AUC=result["roc_auc"], AP=result["pr_auc"],
                            Brier=result["brier"], KS=result["ks"], AUC_low=low, AUC_high=high))
    pd.DataFrame(summary).to_csv(artifact(cfg, "tables/model_metrics.csv"), index=False)

    predictions = {m: pd.read_parquet(artifact(cfg, f"predictions/{m}_{split}.parquet")).set_index("loan_id")
                   for m in models}
    plots(cfg, predictions, "comparison")
    score = predictions["woe_scorecard"]
    rows = []
    for name in ("logreg_baseline", "grade_only"):
        other = predictions[name].loc[score.index]
        assert other.y_true.equals(score.y_true) and score.index.is_unique
        result = compare_models(score.y_true, score.y_prob, other.y_prob, cfg["bootstrap_iterations"], cfg["seed"])
        for test, metrics in [("delong", {"roc_auc": result["delong"]}), ("paired_bootstrap", result["paired_bootstrap"])]:
            for metric, values in metrics.items():
                lo, hi = values["ci95"]
                rows.append(dict(split=split, model_a="woe_scorecard", model_b=name, test=test, metric=metric,
                                 difference=values["difference"], ci_low=lo, ci_high=hi,
                                 z=values.get("z"), p_value=values["p_value"]))
    pd.DataFrame(rows).to_csv(artifact(cfg, "tables/model_comparisons.csv"), index=False)


if __name__ == "__main__":
    run(cli_config())
