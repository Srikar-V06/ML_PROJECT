import argparse
from time import perf_counter
import joblib

import matplotlib.pyplot as plt
import pandas as pd
from threadpoolctl import threadpool_limits

from src.audit.schema import classify
from src.data.splits import load
from src.evaluation.metrics import evaluate
from src.evaluation.plots import plots
from src.models.baselines import model_frame
from src.models.registry import get_model, list_models
from src.utils.io import artifact, config, write_json
from src.utils.logging import log
from src.utils.seed import set_seed


def run(name, cfg):
    set_seed(cfg["seed"])
    splits = cfg.get("evaluation_splits", ["val", "test"])
    frames = load(cfg, names=["train", *splits])
    audit = pd.read_csv(artifact(cfg, "feature_audit.csv"))
    cols = audit.loc[audit.decision.eq("keep"), "feature_name"].tolist()
    assert len(cols) >= 50 and all(classify(c)[0] == "origination_time" for c in cols)
    missing = pd.read_csv(artifact(cfg, "missing_value_table.csv"))
    flags = missing.loc[missing.add_missing_indicator & missing.feature_name.isin(cols), "feature_name"].tolist()
    x = model_frame(frames["train"][cols])
    model = get_model(name)(x, flags, cfg)
    with threadpool_limits(limits=2):
        start = perf_counter()
        model.fit(x, frames["train"].target)
        fit_seconds = perf_counter() - start
    model_path = artifact(cfg, f"models/{name}.joblib")
    joblib.dump(model, model_path, compress=3)
    metrics, thresholds = {}, None
    for split in splits:
        frame = frames[split]
        features = model_frame(frame[cols])
        start = perf_counter()
        p = model.predict_proba(features)[:, 1]
        inference_seconds = perf_counter() - start
        pred = pd.DataFrame(dict(loan_id=frame.index, y_true=frame.target.to_numpy(), y_prob=p, split=split))
        pred.to_parquet(artifact(cfg, f"predictions/{name}_{split}.parquet"), index=False)
        metrics[split] = evaluate(pred.y_true, p, split, name, None, cfg["cost_ratios"], thresholds,
                                  cfg["bootstrap_iterations"], cfg["seed"])
        thresholds = metrics["val"]["thresholds"]
        metrics[split].update(fit_seconds=fit_seconds, inference_seconds_per_1000=inference_seconds*1000/len(p),
                              model_bytes=model_path.stat().st_size)
        if split == splits[-1]:
            plots(cfg, {name: pred}, name)
            if name == "woe_scorecard":
                points = model.points(features)
                for outcome, label in [(0, "Fully paid"), (1, "Default")]:
                    plt.hist(points[pred.y_true == outcome], bins=40, density=True, alpha=.55, label=label)
                plt.xlabel("Credit score (points)")
                plt.ylabel("Density")
                plt.legend()
                plt.tight_layout()
                plt.savefig(artifact(cfg, "plots/scorecard_distribution.png"))
                plt.close()
    write_json(artifact(cfg, f"metrics/{name}.json"), metrics)
    log(cfg, f"train/{name}", len(x), len(cols))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list_models(), required=True)
    parser.add_argument("--config", default="config/base.yaml")
    args = parser.parse_args()
    run(args.model, config(args.config))
