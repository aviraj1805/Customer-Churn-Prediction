import numpy as np
import pytest

from src.config import resolve_path
from src.data import split_X_y
from src.explain import (
    contributions,
    drivers,
    original_field,
    retention_scenarios,
    tenure_outlook,
)
from src.features import ENGINEERED_FEATURES, RAW_FEATURES
from src.models import candidate_pipeline
from src.predict import EXAMPLE_CUSTOMERS, load_model, prepare_customer
from tests.conftest import make_raw_frame

HIGH_RISK = EXAMPLE_CUSTOMERS["New fiber customer, month-to-month, electronic check"]
LOW_RISK = EXAMPLE_CUSTOMERS["Long-standing DSL customer, two-year contract"]
PHONE_ONLY = EXAMPLE_CUSTOMERS["Phone-only customer, one-year contract"]


def train(model_name: str, cfg: dict):
    X, y = split_X_y(make_raw_frame(n_rows=1500, seed=4), cfg)
    pipeline = candidate_pipeline(model_name, cfg)[0]
    if model_name in ("xgboost", "lightgbm"):
        pipeline.set_params(model__n_estimators=40)
    return pipeline.fit(X, y)


@pytest.fixture(scope="module")
def xgb_pipeline():
    from src.config import load_config

    return train("xgboost", load_config())


@pytest.mark.parametrize("model_name", ["xgboost", "lightgbm"])
def test_contributions_add_up_to_the_predicted_probability(model_name, cfg):
    pipeline = train(model_name, cfg)
    X = prepare_customer(HIGH_RISK)
    logit = contributions(pipeline, X).sum(axis=1).iloc[0]
    assert 1 / (1 + np.exp(-logit)) == pytest.approx(pipeline.predict_proba(X)[0, 1], abs=1e-5)


def test_models_without_native_contributions_return_none(cfg):
    assert contributions(train("logistic_regression", cfg), prepare_customer(HIGH_RISK)) is None
    assert drivers(train("logistic_regression", cfg), HIGH_RISK) is None


@pytest.mark.parametrize("column, field", [
    ("categorical__Contract_Two year", "Contract"),
    ("categorical__StreamingTV_No internet service", "StreamingTV"),
    ("categorical__gender_Male", "gender"),
    ("numeric__tenure", "tenure"),
    ("numeric__ChargeIncrease", "ChargeIncrease"),
])
def test_original_field(column, field):
    assert original_field(column) == field


def test_drivers_cover_every_input_once_sorted_by_effect(xgb_pipeline):
    table = drivers(xgb_pipeline, HIGH_RISK)
    assert sorted(table["field"]) == sorted(RAW_FEATURES + ENGINEERED_FEATURES)
    assert table["contribution"].abs().is_monotonic_decreasing
    contrib = contributions(xgb_pipeline, prepare_customer(HIGH_RISK))
    assert table["contribution"].sum() == pytest.approx(contrib.drop(columns="bias").sum(axis=1).iloc[0])
    assert table.set_index("field").loc["Contract", "value"] == "Month-to-month"


def test_retention_scenarios_only_offer_applicable_actions(xgb_pipeline):
    high = retention_scenarios(xgb_pipeline, HIGH_RISK)
    assert {"Move to a one-year contract", "Switch to automatic card payment", "Add tech support"} <= set(high["action"])
    assert high["churn_probability"].between(0, 1).all()
    assert high["change"].is_monotonic_increasing

    low = retention_scenarios(xgb_pipeline, LOW_RISK)
    assert low.empty  # two-year contract, automatic payment, all add-ons already

    phone_only = retention_scenarios(xgb_pipeline, PHONE_ONLY)
    assert "Add tech support" not in set(phone_only["action"])  # no internet service


def test_tenure_outlook_scores_every_month_consistently(xgb_pipeline):
    outlook = tenure_outlook(xgb_pipeline, HIGH_RISK)
    assert outlook["tenure"].tolist() == list(range(1, 73))
    assert outlook["churn_probability"].between(0, 1).all()
    at_two_months = {**HIGH_RISK, "tenure": 2, "TotalCharges": 2 * HIGH_RISK["MonthlyCharges"]}
    expected = xgb_pipeline.predict_proba(prepare_customer(at_two_months))[0, 1]
    assert outlook.loc[outlook["tenure"] == 2, "churn_probability"].iloc[0] == pytest.approx(expected)


@pytest.mark.skipif(not resolve_path("models/best_model.joblib").exists(), reason="no trained model yet")
def test_production_model_explanations_are_plausible():
    pipeline, _ = load_model()
    if drivers(pipeline, HIGH_RISK) is None:
        pytest.skip("deployed model has no native contributions")
    high = drivers(pipeline, HIGH_RISK).set_index("field")["contribution"]
    low = drivers(pipeline, LOW_RISK).set_index("field")["contribution"]
    assert high["Contract"] > 0 > low["Contract"]  # month-to-month raises risk, two-year lowers it
    assert retention_scenarios(pipeline, HIGH_RISK)["change"].iloc[0] < 0
