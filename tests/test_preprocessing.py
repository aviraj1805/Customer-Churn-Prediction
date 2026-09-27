import numpy as np
import pandas as pd
import pytest

from src.data import split_X_y
from src.features import CATEGORICAL_FEATURES, CATEGORY_LEVELS, NUMERIC_FEATURES, add_engineered_features
from src.preprocessing import build_preprocessor, feature_names

# One-hot width: binary columns collapse to a single 0/1 column, the rest keep one column per level.
N_ONE_HOT = sum(1 if len(levels) == 2 else len(levels) for levels in CATEGORY_LEVELS.values())


@pytest.fixture
def X(raw_df, cfg):
    return split_X_y(raw_df, cfg)[0]


@pytest.mark.parametrize("engineered, n_extra", [(False, 0), (True, 3)])
def test_output_is_numeric_with_expected_width(X, engineered, n_extra):
    out = build_preprocessor(engineered).fit_transform(X)
    assert out.shape == (len(X), len(NUMERIC_FEATURES) + n_extra + N_ONE_HOT)
    assert np.isfinite(out).all()


def test_missing_values_are_imputed(X):
    X = X.astype({"TotalCharges": object})
    X.loc[X.index[0], "TotalCharges"] = " "  # blank string, as in the original IBM Telco data
    X.loc[X.index[1], "Contract"] = np.nan
    out = build_preprocessor(engineered_features=True).fit_transform(X)
    assert np.isfinite(out).all()


def test_unseen_category_does_not_break_transform(X):
    pre = build_preprocessor().fit(X)
    new_row = X.head(1).copy()
    new_row["PaymentMethod"] = "Crypto"
    with pytest.warns(UserWarning, match="unknown categories"):
        out = pre.transform(new_row)
    payment_cols = [i for i, n in enumerate(feature_names(pre)) if "PaymentMethod" in n]
    assert out[0, payment_cols].sum() == 0


def test_category_and_object_dtypes_give_identical_output(X):
    """Training reads categoricals as 'category' dtype; the app sends plain strings."""
    as_category = X.astype({col: "category" for col in CATEGORICAL_FEATURES})
    as_object = X.astype({col: object for col in CATEGORICAL_FEATURES})
    pre = build_preprocessor(engineered_features=True).fit(as_category)
    np.testing.assert_allclose(pre.transform(as_category), pre.transform(as_object))


@pytest.mark.parametrize("extra_column", [False, True])
def test_column_order_and_extra_columns_do_not_matter(X, extra_column):
    pre = build_preprocessor(engineered_features=True).fit(X)
    shuffled = X[X.columns[::-1]]
    if extra_column:
        shuffled = shuffled.assign(id=range(len(X)))
    np.testing.assert_allclose(pre.transform(X), pre.transform(shuffled))


def test_engineered_features_values():
    row = pd.DataFrame([{
        "tenure": 10, "MonthlyCharges": 60.0, "TotalCharges": 500.0,
        "OnlineSecurity": "Yes", "OnlineBackup": "No", "DeviceProtection": "Yes",
        "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "No internet service",
    }])
    out = add_engineered_features(row).iloc[0]
    assert out["AvgChargePerMonth"] == pytest.approx(50.0)
    assert out["ChargeIncrease"] == pytest.approx(10.0)
    assert out["NumAddonServices"] == 3
