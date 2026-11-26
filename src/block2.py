"""Train-only model selection and validation handoff for the second project block."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd
from threadpoolctl import threadpool_limits

from src.analysis.block2_eda import eda
from src.analysis.block2_pca import pca_study, save_validation
from src.analysis.block2_search import tune
from src.analysis.block2_trees import decision_diagnostics, forest_diagnostics
from src.audit.schema import CLEAN
from src.data.splits import load
from src.models.baselines import model_frame
from src.models.block2 import BOOSTERS
from src.models.registry import get_model
from src.utils.io import artifact, config


def run(cfg, stage, trials=12, fit_row_limit=20000):
    if stage not in {"all", "classical", "boosters", "eda"}:
        raise ValueError(f"Unknown Block 2 stage: {stage}")
    if trials < 1 or fit_row_limit < 0:
        raise ValueError("Trials must be positive and fit_row_limit must be nonnegative.")
    names = ("train", "val") if stage in {"all", "classical"} else ("train",)
    frames = load(cfg, names=names)
    audit = pd.read_csv(artifact(cfg, "feature_audit.csv"))
    columns = audit.loc[audit.decision.eq("keep"), "feature_name"].tolist()
    if not columns or set(columns) - CLEAN or len(columns) != len(set(columns)):
        raise ValueError("Unexpected fields in the origination audit.")
    ledger = pd.read_csv(artifact(cfg, "missing_value_table.csv")).set_index("feature_name")
    flags = ledger.loc[columns, "add_missing_indicator"]
    if not pd.api.types.is_bool_dtype(flags) or flags.isna().any():
        raise ValueError("Expected boolean missing-indicator decisions for every audited input.")
    flags = flags.index[flags].tolist()
    train = frames["train"]
    X, y = model_frame(train[columns]), train.target
    with threadpool_limits(limits=2):
        if stage in {"all", "boosters"}:
            for name in BOOSTERS:
                tune(name, X, y, cfg, trials, fit_row_limit)
        if stage in {"all", "eda"}:
            eda(X, y, train.issue_d, flags, cfg)
        if stage in {"all", "classical"}:
            V, vy = model_frame(frames["val"][columns]), frames["val"].target
            records, best_params = [], {}
            for name, diagnostics in [("decision_tree", decision_diagnostics),
                                      ("random_forest", forest_diagnostics)]:
                params = tune(name, X, y, cfg, trials, fit_row_limit)
                best_params[name] = params
                model = get_model(name)(X, flags, cfg)
                model.named_steps["model"].set_params(**params)
                model.fit(X, y)
                records.append(save_validation(name, model, vy, model.predict_proba(V)[:, 1], cfg))
                diagnostics(model, (X, y), (V, vy), cfg)
            pd.DataFrame(records).to_csv(artifact(cfg, "tables/block2_classical.csv"), index=False)
            pca_rows = cfg.get("pca_fit_rows", len(X))
            pca_x = X.sample(min(pca_rows, len(X)), random_state=cfg["seed"]) if pca_rows < len(X) else X
            pca_study((pca_x, y.loc[pca_x.index]), (V, vy), flags, best_params["random_forest"], cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--sample", type=Path, help="Alternate location for P1's frozen sample")
    parser.add_argument("--stage", choices=["all", "boosters", "classical", "eda"], default="all")
    parser.add_argument("--trials", type=int, help="Total completed Optuna trials per model")
    parser.add_argument("--fit-row-limit", type=int,
                        help="Seeded sample per inner fitting fold (0 uses all rows)")
    args = parser.parse_args()
    cfg = config(args.config)
    if args.sample:
        cfg["paths"]["sample"] = str(args.sample)
    trials = cfg.get("search_trials", 12) if args.trials is None else args.trials
    limit = cfg.get("search_fit_rows", 20000) if args.fit_row_limit is None else args.fit_row_limit
    run(cfg, args.stage, trials, limit)
