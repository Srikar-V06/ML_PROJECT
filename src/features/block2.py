"""Underwriting-only features and fold-fitted preprocessing for Block 2."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from src.audit.schema import CLEAN


# (numerator, denominator, multiplier). A zero/negative denominator is undefined.
RATIOS = {
    "loan_to_income": ("loan_amnt", "annual_inc", 1),
    "payment_to_income": ("installment", "annual_inc", 12),
    "revolving_to_income": ("revol_bal", "annual_inc", 1),
    "open_account_share": ("open_acc", "total_acc", 1),
}
GRADE = {"grade", "sub_grade"}
AFFORDABILITY = {"loan_amnt", "term", "installment", "int_rate", "annual_inc",
                 "annual_inc_joint", "dti", "dti_joint"}
APPLICATION = {"emp_title", "emp_length", "home_ownership", "verification_status",
               "verification_status_joint", "purpose", "addr_state", "initial_list_status",
               "application_type"}
FEATURE_GROUPS = {
    "lc_grade": GRADE, "affordability": AFFORDABILITY, "application": APPLICATION,
    "bureau": CLEAN - GRADE - AFFORDABILITY - APPLICATION, "engineered": set(RATIOS),
}


class UnderwritingFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, engineered=True):
        self.engineered = engineered

    def fit(self, X, y=None):
        unknown = set(X.columns) - CLEAN
        if unknown or not X.columns.is_unique:
            raise ValueError(f"Expected unique, audited origination fields. Unreviewed: {sorted(unknown)}")
        self.feature_names_in_ = X.columns.to_numpy(copy=True)
        self.ratios_ = {key: spec for key, spec in RATIOS.items()
                        if self.engineered and set(spec[:2]).issubset(X.columns)}
        return self

    def transform(self, X):
        check_is_fitted(self, "ratios_")
        result = X.loc[:, self.feature_names_in_].copy()
        for name, (numerator, denominator, multiplier) in self.ratios_.items():
            positive = result[denominator].where(result[denominator] > 0)
            result[name] = result[numerator].mul(multiplier).div(positive)
        return result

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "ratios_")
        return np.asarray([*self.feature_names_in_, *self.ratios_], dtype=object)


def build_preprocessor(kind, flags=(), *, engineered=True, impute_tree=False):
    """Consume P1 model_frame output. Pass flags=() for the missing-indicator ablation."""
    if kind not in {"tree", "linear"}:
        raise ValueError("kind must be 'tree' or 'linear'")
    numeric = "passthrough"
    if kind == "linear" or impute_tree:
        numeric = SimpleImputer(strategy="median", keep_empty_features=True)
        if kind == "linear":
            numeric = make_pipeline(numeric, StandardScaler())
    def categories(limit=None):
        return make_pipeline(
            SimpleImputer(strategy="constant", fill_value="__missing__", keep_empty_features=True),
            OneHotEncoder(handle_unknown="ignore", max_categories=limit, sparse_output=False),
        )

    # Bound employment-title width while keeping grades, states and other ordinary levels distinct.
    transforms = [
        ("numeric", numeric, make_column_selector(dtype_include=np.number)),
        ("category", categories(), make_column_selector(pattern=r"^(?!emp_title$)", dtype_exclude=np.number)),
        ("title", categories(32), make_column_selector(pattern=r"^emp_title$", dtype_exclude=np.number)),
    ]
    if len(flags):
        transforms.append(("missing", MissingIndicator(features="all"), list(flags)))
    return Pipeline([("features", UnderwritingFeatures(engineered)),
                     ("columns", ColumnTransformer(transforms))])
