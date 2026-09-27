"""Tests for the app's event handlers (the UI itself is not launched)."""
import pandas as pd
import pytest

from src.config import resolve_path

pytestmark = pytest.mark.skipif(
    not resolve_path("models/best_model.joblib").exists(), reason="the app needs a trained model"
)

HIGH_RISK = "New fiber customer, month-to-month, electronic check"
LOW_RISK = "Long-standing DSL customer, two-year contract"


@pytest.fixture(scope="module")
def app_module():
    pytest.importorskip("gradio")
    from app import app

    return app


def test_analyse_returns_every_predict_tab_output(app_module):
    gauge, verdict, driver_bars, scenarios, plot_data, note = app_module.analyse(
        *app_module.to_ui_values(app_module.EXAMPLE_CUSTOMERS[HIGH_RISK]))
    assert "churn probability" in gauge
    assert "flag for a retention offer" in verdict
    assert "Contract" in driver_bars
    assert "two-year contract" in scenarios
    assert set(plot_data["series"]) == {"This customer", "Decision threshold"}
    assert plot_data["churn_probability"].between(0, 100).all()
    assert "predicted risk goes from" in note


@pytest.mark.parametrize("missing_total", [None, 0])
def test_missing_total_charges_is_estimated_from_tenure(app_module, missing_total):
    customer = app_module.EXAMPLE_CUSTOMERS[LOW_RISK]
    total_idx = app_module.RAW_FEATURES.index("TotalCharges")
    explicit = app_module.to_ui_values(customer)
    explicit[total_idx] = customer["tenure"] * customer["MonthlyCharges"]
    missing = app_module.to_ui_values(customer)
    missing[total_idx] = missing_total
    assert app_module.analyse(*missing)[0] == app_module.analyse(*explicit)[0]
    assert "estimated" in app_module.analyse(*missing)[-1]


def test_dependent_options_follow_internet_and_phone_service(app_module):
    no_internet = app_module.dependent_update("TechSupport", "No", "Yes")
    assert no_internet["choices"] == ["No internet service"] and no_internet["interactive"] is False
    assert no_internet["value"] == "No internet service"
    with_internet = app_module.dependent_update("TechSupport", "DSL", "No internet service")
    assert with_internet["value"] == "No" and with_internet["interactive"] is True
    assert app_module.dependent_choices("MultipleLines", "No") == ["No phone service"]


def test_invalid_input_raises_a_gradio_error(app_module):
    import gradio as gr

    values = app_module.to_ui_values(app_module.EXAMPLE_CUSTOMERS[LOW_RISK])
    values[app_module.RAW_FEATURES.index("Contract")] = "Weekly"
    with pytest.raises(gr.Error):
        app_module.analyse(*values)


def test_score_batch_on_the_sample_file(app_module):
    summary, tiers, top, download = app_module.score_batch(str(app_module.SAMPLE_CSV))
    assert "Customers scored" in summary and ">40<" in summary
    assert tiers["customers"].sum() == 40 and list(tiers["risk_tier"]) == app_module.RISK_TIERS
    assert len(top) == 25 and top["Churn probability (%)"].is_monotonic_decreasing
    scored = pd.read_csv(download["value"])
    assert len(scored) == 40 and {"id", "churn_probability", "will_churn", "risk_tier"} <= set(scored.columns)


def test_score_batch_rejects_bad_files(app_module, tmp_path, monkeypatch):
    import gradio as gr

    sample = pd.read_csv(app_module.SAMPLE_CSV)
    sample.drop(columns="Contract").to_csv(tmp_path / "missing.csv", index=False)
    with pytest.raises(gr.Error, match="Contract"):
        app_module.score_batch(str(tmp_path / "missing.csv"))

    monkeypatch.setattr(app_module, "MAX_BATCH_ROWS", 10)
    with pytest.raises(gr.Error, match="at most 10 rows"):
        app_module.score_batch(str(app_module.SAMPLE_CSV))


def test_score_batch_scores_rows_with_unknown_values(app_module, tmp_path):
    sample = pd.read_csv(app_module.SAMPLE_CSV)
    sample.loc[0, "PaymentMethod"] = "Crypto"
    sample.to_csv(tmp_path / "unknown.csv", index=False)
    summary, tiers, _, _ = app_module.score_batch(str(tmp_path / "unknown.csv"))
    assert tiers["customers"].sum() == 40
    assert app_module.score_batch(None)[0] == ""


def test_score_batch_without_id_column(app_module, tmp_path):
    pd.read_csv(app_module.SAMPLE_CSV).drop(columns="id").to_csv(tmp_path / "no_id.csv", index=False)
    _, _, top, download = app_module.score_batch(str(tmp_path / "no_id.csv"))
    assert "ID" not in top.columns and len(pd.read_csv(download["value"])) == 40


def test_threshold_view_uses_the_nearest_saved_threshold(app_module):
    cards, matrix, business = app_module.threshold_view(0.362)
    row = app_module.ART.thresholds.set_index("threshold").loc[0.36]
    assert f"{int(row['tp']):,}" in matrix and f"{int(row['tn']):,}" in matrix
    assert app_module.ui.pct(row["recall"]) in cards
    assert "Out of <b>10,000</b>" in business
    low_cards, _, _ = app_module.threshold_view(0.1)
    assert low_cards != cards


def test_performance_tables_cover_every_model(app_module):
    table = app_module.comparison_display()
    assert len(table) == 6 and table["Model"].str.contains("(deployed)", regex=False).sum() == 1
    roc = app_module.curve_data("roc")
    assert {"Random guess", "XGBoost (tuned)"} <= set(roc["model"])
    assert set(roc["model"]) <= set(app_module.ui.MODEL_COLORS)
    tradeoff = app_module.tradeoff_data()
    assert set(tradeoff["metric"]) == {"Precision", "Recall", "F1"} and tradeoff["value"].between(0, 100).all()
    assert "Decision threshold" in app_module.model_details_md()


def test_precision_recall_view_has_a_random_guess_baseline_at_the_churn_rate(app_module):
    pr = app_module.curve_data("pr")
    baseline = pr[pr["model"] == "Random guess"]["y"]
    assert baseline.tolist() == pytest.approx([app_module.ART.churn_rate] * 2)
    assert set(app_module.CURVE_AXES) == {"roc", "pr"}


def test_segment_view_and_findings_come_from_the_saved_rates(app_module):
    plot, note = app_module.segment_view("PaymentMethod")
    data = plot["value"]
    assert set(data["level"]) == set(app_module.CATEGORY_LEVELS["PaymentMethod"])
    assert plot["x_title"] == "Payment method" and "Electronic check" in note
    findings = app_module.key_findings_md()
    assert "Month-to-month customers churn at 42.1%" in findings and "Senior citizens churn at 50.0%" in findings


def test_json_api_returns_serialisable_prediction_with_drivers(app_module):
    import json

    import gradio as gr

    customer = {k: v for k, v in app_module.EXAMPLE_API_CUSTOMER.items() if k != "TotalCharges"}
    result = app_module.predict_api(customer)  # TotalCharges left out -> estimated
    assert json.loads(json.dumps(result)) == result
    assert result["risk_tier"] == "Very high" and result["will_churn"] is True
    assert len(result["top_drivers"]) == 5 and result["top_drivers"][0]["field"] in app_module.RAW_FEATURES
    with pytest.raises(gr.Error, match="Contract"):
        app_module.predict_api({**customer, "Contract": "Weekly"})
    with pytest.raises(gr.Error):
        app_module.predict_api({"tenure": "abc", "MonthlyCharges": 20})


def test_about_page_documents_the_api_and_model_card(app_module):
    md = app_module.ui.about_md(app_module.ART.metadata, app_module.ART.importance, app_module.EXAMPLE_API_CUSTOMER)
    assert "Model card" in md and "Limitations" in md and "Fairness check" in md
    assert 'api_name="/predict"' in md and "gradio_api/call/predict" in md
