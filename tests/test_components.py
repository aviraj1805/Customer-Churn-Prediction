import pandas as pd
import pytest

from app import components as ui

pytest.importorskip("gradio")


def test_format_value_is_human_readable():
    assert ui.format_value("MonthlyCharges", 85.5) == "$85.50"
    assert ui.format_value("ChargeIncrease", -3.2) == "-$3.20"
    assert ui.format_value("SeniorCitizen", 1) == "Yes"
    assert ui.format_value("tenure", 12.0) == "12"
    assert ui.format_value("Contract", "Two year") == "Two year"


def test_gauge_shows_probability_threshold_and_tier_color():
    html = ui.gauge_html(0.8, threshold=0.4)
    assert "80.0%" in html and "threshold 40%" in html
    assert "var(--risk-very-high)" in html
    assert "Low risk" not in html


def test_verdict_flags_only_above_threshold():
    assert "flag for a retention offer" in ui.verdict_html(0.5, 0.4, 0.2)
    assert "no action needed" in ui.verdict_html(0.3, 0.4, 0.2)
    assert "2.5x the average" in ui.verdict_html(0.5, 0.4, 0.2)


def test_drivers_html_uses_direction_and_escapes_values():
    table = pd.DataFrame({"field": ["Contract", "tenure"], "value": ["<b>x</b>", 60.0], "contribution": [0.9, -0.3]})
    html = ui.drivers_html(table)
    assert 'class="bar up"' in html and 'class="bar down"' in html
    assert "<b>x</b>" not in html and "&lt;b&gt;x&lt;/b&gt;" in html
    assert "boosting models only" in ui.drivers_html(None)


def test_scenarios_html_handles_empty_and_changes():
    assert "already has the lowest-risk" in ui.scenarios_html(pd.DataFrame(columns=["action"]), 0.2)
    table = pd.DataFrame({"action": ["Move to a two-year contract"], "churn_probability": [0.3], "change": [-0.5]})
    html = ui.scenarios_html(table, 0.8)
    assert "-50.0 pts" in html and "80.0%" in html and "30.0%" in html


def test_confusion_and_business_views_use_the_counts():
    row = pd.Series({"tp": 80, "fp": 20, "fn": 20, "tn": 880, "recall": 0.8})
    cm = ui.confusion_html(row)
    assert "880" in cm and "caught churner" in cm
    business = ui.business_html(row, per=10_000)
    assert "<b>1,000</b> of whom will churn" in business  # 100 churners in 1,000 rows -> 1,000 per 10,000
    assert "flag <b>1,000</b>" in business and "catch <b>800</b>" in business


def test_kpi_cards_and_importance_bars():
    assert ui.kpi_cards_html([("Recall", "78%", "of churners")]).count('class="kpi"') == 1
    importance = pd.DataFrame({"feature": ["Contract", "gender"], "importance": [0.06, -0.001]})
    html = ui.importance_html(importance)
    assert "width:100.0%" in html and "width:0.0%" in html
