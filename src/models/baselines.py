import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.models.registry import register_model


def model_frame(frame):
    result = frame.copy()
    for col in result:
        s = result[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            result[col] = (s - pd.Timestamp("1970-01-01")).dt.days.astype(float)
        elif pd.api.types.is_numeric_dtype(s):
            result[col] = s.to_numpy(dtype=float, na_value=np.nan)
        else:
            result[col] = s.astype(object).where(s.notna(), np.nan)
    return result


def preprocessor(X, flags):
    num = X.select_dtypes(include="number").columns.tolist()
    cat = X.columns.difference(num).tolist()
    transforms = [
        ("numeric", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()), num),
        ("categories", make_pipeline(SimpleImputer(strategy="constant", fill_value="__missing__", keep_empty_features=True),
                                     OneHotEncoder(handle_unknown="ignore", min_frequency=25)), cat)]
    if flags:
        transforms.append(("missing", MissingIndicator(features="all"), flags))
    return ColumnTransformer(transforms, sparse_threshold=1.0)


@register_model("constant")
def constant(X, flags, cfg):
    return DummyClassifier(strategy="prior")


@register_model("grade_only")
def grade_only(X, flags, cfg):
    cols = ["grade", "sub_grade"]
    return make_pipeline(ColumnTransformer([("grades", OneHotEncoder(handle_unknown="ignore"), cols)]),
                         LogisticRegression(max_iter=300, tol=1e-6))


@register_model("logreg_baseline")
def logistic(X, flags, cfg):
    return Pipeline([("preprocess", preprocessor(X, flags)), ("logistic", LogisticRegression(max_iter=1000,
                     tol=1e-5, random_state=cfg["seed"]))])


def sanity(X, y, seed):
    slice_x, slice_y = X.iloc[:1000], y.iloc[:1000]
    tree = make_pipeline(preprocessor(slice_x, []), DecisionTreeClassifier(random_state=seed))
    tree.fit(slice_x, slice_y)
    auc = float(roc_auc_score(slice_y, tree.predict_proba(slice_x)[:, 1]))
    assert auc > .99, f"Memorization failed: {auc}"
    return {"rows": len(slice_y), "train_auc": auc}
