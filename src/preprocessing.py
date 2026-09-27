"""Preprocessing: one scikit-learn object that turns raw customer rows into model-ready numbers.

It is fitted on training data only and saved inside the model pipeline, so training, evaluation,
batch prediction and the web app all apply exactly the same steps. This replaces the notebook's
manual Yes/No mapping, ``get_dummies`` and train/test column alignment.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.features import (
    CATEGORICAL_FEATURES,
    ENGINEERED_FEATURES,
    NUMERIC_FEATURES,
    add_engineered_features,
)


def coerce_numeric(X: pd.DataFrame) -> pd.DataFrame:
    """Force numeric columns to numbers; blanks such as ' ' in TotalCharges become NaN for the imputer."""
    X = X.copy()
    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


def build_preprocessor(engineered_features: bool = False) -> Pipeline:
    """Build the (unfitted) preprocessing pipeline.

    Numeric columns: median imputation, then standard scaling (needed by Logistic Regression,
    harmless for tree models). Categorical columns: most-frequent imputation, then one-hot
    encoding. ``drop="if_binary"`` turns Yes/No columns into a single 0/1 column, like the
    original notebook, and ``handle_unknown="ignore"`` keeps inference working if a new
    category ever appears.
    """
    numeric_columns = NUMERIC_FEATURES + (ENGINEERED_FEATURES if engineered_features else [])
    numeric_steps = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_steps = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False)),
    ])
    encode = ColumnTransformer([
        ("numeric", numeric_steps, numeric_columns),
        ("categorical", categorical_steps, CATEGORICAL_FEATURES),
    ])

    # The FunctionTransformers deliberately do not declare output names: when they do, sklearn
    # relabels columns positionally, which silently scrambles inputs that arrive in another order.
    # The ColumnTransformer selects columns by name, so input column order never matters.
    steps = [("coerce", FunctionTransformer(coerce_numeric))]
    if engineered_features:
        steps.append(("engineer", FunctionTransformer(add_engineered_features)))
    steps.append(("encode", encode))
    return Pipeline(steps)


def feature_names(preprocessor: Pipeline) -> list[str]:
    """Names of the model-ready columns, e.g. 'numeric__tenure' or 'categorical__Contract_Two year'."""
    return list(preprocessor.named_steps["encode"].get_feature_names_out())
