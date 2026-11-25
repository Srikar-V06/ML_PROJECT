"""Registry factories and the search spaces shared with the next training block."""
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.features.block2 import build_preprocessor
from src.models.registry import register_model


BOOSTERS = ("adaboost", "gradient_boosting", "xgboost", "lightgbm", "catboost")


def estimator(name, seed):
    if name == "decision_tree":
        return DecisionTreeClassifier(random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=2)
    if name == "adaboost":
        return AdaBoostClassifier(random_state=seed)
    if name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=seed)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(random_state=seed, tree_method="hist", n_jobs=2)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=seed, n_jobs=2, verbosity=-1, deterministic=True,
                              force_col_wise=True)
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(random_seed=seed, thread_count=2, verbose=False,
                                  allow_writing_files=False)
    raise KeyError(name)


def model_factory(name):
    def factory(X, flags, cfg):
        model = estimator(name, cfg["seed"]).set_params(**cfg.get("model_params", {}).get(name, {}))
        preprocess = build_preprocessor("tree", flags, impute_tree=name in BOOSTERS[:2])
        return Pipeline([("preprocess", preprocess), ("model", model)])
    return factory


for model_name in ("decision_tree", "random_forest", *BOOSTERS):
    register_model(model_name)(model_factory(model_name))


def search_space(trial, name):
    if name in {"decision_tree", "random_forest"}:
        params = dict(max_depth=trial.suggest_int("max_depth", 3, 16),
                      min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, 128, log=True))
        if name == "decision_tree":
            params["ccp_alpha"] = trial.suggest_categorical("ccp_alpha", [0., .00001, .0001, .001])
        else:
            params["max_features"] = trial.suggest_categorical("max_features", ["sqrt", .5])
            params["n_estimators"] = trial.suggest_categorical("n_estimators", [64, 128, 256])
        return params
    rounds = "iterations" if name == "catboost" else "n_estimators"
    params = {rounds: trial.suggest_int(rounds, 50, 250, step=50),
              "learning_rate": trial.suggest_float("learning_rate", .02, 1. if name == "adaboost" else .2, log=True)}
    if name == "lightgbm":
        params["num_leaves"] = trial.suggest_int("num_leaves", 8, 64)
    elif name != "adaboost":
        depth = "depth" if name == "catboost" else "max_depth"
        params[depth] = trial.suggest_int(depth, 2, 7)
    return params
