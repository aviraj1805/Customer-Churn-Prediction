import copy
import json

import joblib
import pandas as pd
import pytest

from src.config import resolve_path
from src.data import split_X_y
from src.models import candidate_pipeline
from src.predict import (
    EXAMPLE_CUSTOMERS,
    load_model,
    predict_customer,
    prepare_customer,
    risk_tier,
    score_file,
    unknown_categories,
    write_submission,
)
from tests.conftest import make_raw_frame

HIGH_RISK = EXAMPLE_CUSTOMERS["New fiber customer, month-to-month, electronic check"]
LOW_RISK = EXAMPLE_CUSTOMERS["Long-standing DSL customer, two-year contract"]
PHONE_ONLY = EXAMPLE_CUSTOMERS["Phone-only customer, one-year contract"]


@pytest.fixture(scope="module")
def saved_model(tmp_path_factory):
    """A small pipeline trained on synthetic data and saved like src.train does."""
    from src.config import load_config

    cfg = load_config()
    X, y = split_X_y(make_raw_frame(n_rows=1500, seed=1), cfg)
    pipeline = candidate_pipeline("logistic_regression", cfg)[0].fit(X, y)
    folder = tmp_path_factory.mktemp("model")
    joblib.dump(pipeline, folder / "model.joblib")
    (folder / "metadata.json").write_text(json.dumps({"threshold": 0.4, "display_name": "Test model"}))
    return folder / "model.joblib", folder / "metadata.json"


def test_load_model_restores_pipeline_and_metadata(saved_model):
    pipeline, metadata = load_model(*saved_model)
    assert hasattr(pipeline, "predict_proba")
    assert metadata["threshold"] == 0.4


def test_predict_customer_returns_probability_and_decision(saved_model):
    pipeline, metadata = load_model(*saved_model)
    result = predict_customer(HIGH_RISK, pipeline, metadata["threshold"])
    assert 0 <= result["churn_probability"] <= 1
    assert result["will_churn"] == (result["churn_probability"] >= 0.4)


def test_risky_profile_scores_higher_than_loyal_profile(saved_model):
    pipeline, _ = load_model(*saved_model)
    high = predict_customer(HIGH_RISK, pipeline, 0.5)["churn_probability"]
    low = predict_customer(LOW_RISK, pipeline, 0.5)["churn_probability"]
    assert high > low


def test_prepare_customer_rejects_missing_field():
    customer = {k: v for k, v in HIGH_RISK.items() if k != "Contract"}
    with pytest.raises(ValueError, match="Contract"):
        prepare_customer(customer)


@pytest.mark.parametrize("field, value", [("Contract", "Weekly"), ("tenure", -1), ("SeniorCitizen", 2)])
def test_prepare_customer_rejects_invalid_values(field, value):
    with pytest.raises(ValueError, match=field):
        prepare_customer({**HIGH_RISK, field: value})


def test_prepare_customer_resolves_contradictory_services():
    no_internet = prepare_customer({**HIGH_RISK, "InternetService": "No"}).iloc[0]
    assert no_internet["OnlineSecurity"] == "No internet service"
    assert no_internet["StreamingTV"] == "No internet service"

    no_phone = prepare_customer({**HIGH_RISK, "PhoneService": "No", "MultipleLines": "Yes"}).iloc[0]
    assert no_phone["MultipleLines"] == "No phone service"

    internet_again = prepare_customer({**PHONE_ONLY, "InternetService": "DSL"}).iloc[0]
    assert internet_again["TechSupport"] == "No"


def test_score_file_keeps_ids(saved_model, tmp_path):
    pipeline, _ = load_model(*saved_model)
    raw = make_raw_frame(n_rows=50, seed=2, with_target=False)
    raw.to_csv(tmp_path / "in.csv", index=False)
    scored = score_file(tmp_path / "in.csv", tmp_path / "out.csv", pipeline, threshold=0.4)
    assert scored["id"].tolist() == raw["id"].tolist()
    assert scored["churn_probability"].between(0, 1).all()


def test_write_submission_follows_sample_order(saved_model, tmp_path, cfg):
    pipeline, _ = load_model(*saved_model)
    test = make_raw_frame(n_rows=40, seed=3, with_target=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    template = test[["id"]].sample(frac=1, random_state=0).assign(Churn=0.5)
    template.to_csv(tmp_path / "sample_submission.csv", index=False)

    cfg = copy.deepcopy(cfg)
    cfg["paths"].update(raw_test=tmp_path / "test.csv", sample_submission=tmp_path / "sample_submission.csv",
                        submission=tmp_path / "submission.csv")
    submission = write_submission(cfg, pipeline)
    assert submission["id"].tolist() == template["id"].tolist()
    assert pd.read_csv(tmp_path / "submission.csv")["Churn"].between(0, 1).all()


@pytest.mark.skipif(not resolve_path("models/best_model.joblib").exists(), reason="no trained model yet")
def test_saved_production_model_scores_examples():
    pipeline, metadata = load_model()
    probas = {name: predict_customer(c, pipeline, metadata["threshold"])["churn_probability"]
              for name, c in EXAMPLE_CUSTOMERS.items()}
    assert probas["New fiber customer, month-to-month, electronic check"] > 0.5
    assert probas["Long-standing DSL customer, two-year contract"] < 0.1


@pytest.mark.parametrize("proba, tier", [
    (0.0, "Low"), (0.199, "Low"), (0.2, "Moderate"), (0.399, "Moderate"),
    (0.4, "High"), (0.699, "High"), (0.7, "Very high"), (1.0, "Very high"),
])
def test_risk_tiers_are_anchored_on_the_threshold(proba, tier):
    assert risk_tier(proba, threshold=0.4) == tier


def test_only_high_tiers_are_flagged(saved_model):
    pipeline, _ = load_model(*saved_model)
    for customer in EXAMPLE_CUSTOMERS.values():
        result = predict_customer(customer, pipeline, 0.4)
        assert result["will_churn"] == (result["risk_tier"] in ("High", "Very high"))


def test_unknown_categories_are_counted_per_column():
    df = make_raw_frame(n_rows=20, seed=5, with_target=False)
    df.loc[:2, "Contract"] = "Weekly"
    df.loc[0, "PaymentMethod"] = "Crypto"
    assert unknown_categories(df) == {"Contract": 3, "PaymentMethod": 1}
    assert unknown_categories(make_raw_frame(n_rows=20, seed=5, with_target=False)) == {}
