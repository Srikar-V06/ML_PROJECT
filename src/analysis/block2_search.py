"""Resumable Optuna selection on the four frozen expanding Train folds."""
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.models.block2 import search_space
from src.models.registry import get_model
from src.utils.io import artifact, digest, write_json


def tune(name, X, y, cfg, trials, row_limit):
    root = Path(cfg["paths"]["splits"])
    membership = pd.read_parquet(root / "train_expanding_cv.parquet")
    folds = []
    for _, group in membership.groupby("fold", sort=True):
        fit = group.loc[group.role.eq("train"), "id"]
        valid = group.loc[group.role.eq("val"), "id"]
        if row_limit and len(fit) > row_limit:
            fit = fit.sample(n=row_limit, random_state=cfg["seed"])
        fit_x = X.loc[fit]
        # Repeat P1's indicator rule on the fold's fitting rows.
        missing = fit_x.isna().mean()
        flags = missing.index[(missing > .005) & (missing < 1)].tolist()
        folds.append((fit, valid, flags))

    def objective(trial):
        params = search_space(trial, name)
        scores = []
        for fit, valid, flags in folds:
            model = get_model(name)(X, flags, cfg)
            model.named_steps["model"].set_params(**params)
            model.fit(X.loc[fit], y.loc[fit])
            pred = model.predict_proba(X.loc[valid])[:, 1]
            scores.append(float(roc_auc_score(y.loc[valid], pred)))
        trial.set_user_attr("fold_auc", scores)
        return float(np.mean(scores))

    database = artifact(cfg, "block2/search.sqlite3").resolve().as_posix()
    study_name = f"v1-{name}-{digest(root / 'manifest.json')}-{cfg['seed']}-{row_limit}"
    study = optuna.create_study(study_name=study_name, storage=f"sqlite:///{database}",
                                direction="maximize", load_if_exists=True)
    completed = len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)))
    if completed < trials:
        # Vary the sampler seed across restarted batches to avoid replaying the startup trials.
        study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{database}",
                                  sampler=optuna.samplers.TPESampler(seed=cfg["seed"] + len(study.trials)))
        study.optimize(objective, n_trials=trials - completed)
    best = study.best_trial
    write_json(artifact(cfg, f"block2/search_{name}.json"),
               dict(study=study_name, params=best.params, mean_auc=best.value,
                    fold_auc=best.user_attrs["fold_auc"], fit_row_limit=row_limit,
                    completed_trials=len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)))))
    return best.params
