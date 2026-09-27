"""Tests for the Gradio app's prediction function (the UI itself is not launched)."""
import pytest

from src.config import resolve_path

pytestmark = pytest.mark.skipif(
    not resolve_path("models/best_model.joblib").exists(), reason="the app needs a trained model"
)


@pytest.fixture(scope="module")
def app_module():
    gr = pytest.importorskip("gradio")
    from app import app

    return app, gr


def test_predict_returns_label_probabilities_and_explanation(app_module):
    app, _ = app_module
    customer = app.EXAMPLE_CUSTOMERS["New fiber customer, month-to-month, electronic check"]
    label, explanation = app.predict(*app.to_ui_values(customer))
    assert label["Churn"] + label["No churn"] == pytest.approx(1)
    assert "Month-to-month contract" in explanation


@pytest.mark.parametrize("missing_total", [None, 0])
def test_missing_total_charges_is_estimated_from_tenure(app_module, missing_total):
    app, _ = app_module
    customer = app.EXAMPLE_CUSTOMERS["Long-standing DSL customer, two-year contract"]
    total_idx = app.RAW_FEATURES.index("TotalCharges")

    estimated = app.to_ui_values(customer)
    estimated[total_idx] = customer["tenure"] * customer["MonthlyCharges"]
    missing = app.to_ui_values(customer)
    missing[total_idx] = missing_total

    assert app.predict(*missing)[0] == pytest.approx(app.predict(*estimated)[0])


def test_invalid_input_raises_a_gradio_error(app_module):
    app, gr = app_module
    values = app.to_ui_values(app.EXAMPLE_CUSTOMERS["Phone-only customer, one-year contract"])
    values[app.RAW_FEATURES.index("Contract")] = "Weekly"
    with pytest.raises(gr.Error):
        app.predict(*values)
