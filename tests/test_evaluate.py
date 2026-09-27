import numpy as np
import pandas as pd
import pytest

from src.data import split_X_y
from src.evaluate import (
    choose_threshold,
    classification_metrics,
    curve_points,
    segment_churn_rates,
    threshold_table,
)
from tests.conftest import make_raw_frame

RNG = np.random.default_rng(0)
Y_TRUE = RNG.integers(0, 2, 2000)
PROBA = np.clip(0.3 * Y_TRUE + RNG.uniform(0, 0.7, 2000), 0, 1)


def test_choose_threshold_maximises_f1():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.6, 0.4, 0.7, 0.8, 0.9])
    # Cut-off 0.4 flags {0.4, 0.6, 0.7, 0.8, 0.9}: 4 of 4 churners, 1 false alarm -> F1 = 8/9, the best possible.
    assert choose_threshold(y_true, proba) == pytest.approx(0.4)


def test_classification_metrics_on_perfect_ranking():
    y_true = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.3, 0.6, 0.9])
    m = classification_metrics(y_true, proba, threshold=0.5)
    assert m == pytest.approx({"roc_auc": 1, "pr_auc": 1, "accuracy": 1, "precision": 1, "recall": 1, "f1": 1})


def test_threshold_changes_precision_recall_but_not_auc():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    proba = np.array([0.1, 0.4, 0.55, 0.5, 0.7, 0.9])
    low, high = classification_metrics(y_true, proba, 0.3), classification_metrics(y_true, proba, 0.6)
    assert low["roc_auc"] == high["roc_auc"]
    assert low["recall"] > high["recall"]
    assert low["precision"] < high["precision"]


def test_threshold_table_matches_metrics_and_is_monotonic():
    table = threshold_table(Y_TRUE, PROBA)
    assert (table[["tp", "fp", "fn", "tn"]].sum(axis=1) == len(Y_TRUE)).all()
    assert table["recall"].is_monotonic_decreasing
    assert table["flagged_rate"].is_monotonic_decreasing
    row = table.loc[table["threshold"] == 0.5].iloc[0]
    expected = classification_metrics(Y_TRUE, PROBA, 0.5)
    for metric in ("precision", "recall", "f1", "accuracy"):
        assert row[metric] == pytest.approx(expected[metric])


def test_curve_points_cover_every_model_on_a_fixed_grid():
    points = curve_points(Y_TRUE, {"xgboost": PROBA, "baseline": PROBA[::-1]}, n_points=11)
    assert len(points) == 2 * 2 * 11
    assert points["y"].between(0, 1).all()
    roc = points[(points["curve"] == "roc") & (points["model"] == "XGBoost (tuned)")]
    assert roc["y"].iloc[-1] == pytest.approx(1.0)
    assert roc["y"].is_monotonic_increasing


def test_segment_churn_rates_match_a_manual_groupby(cfg):
    X, y = split_X_y(make_raw_frame(n_rows=600), cfg)
    table = segment_churn_rates(X, y)
    contract = table[table["feature"] == "Contract"].set_index("level")
    manual = pd.Series(y.to_numpy()).groupby(X["Contract"].to_numpy()).mean()
    for level, rate in manual.items():
        assert contract.loc[level, "churn_rate"] == pytest.approx(rate)
    assert contract["customers"].sum() == len(X)
    tenure = table[table["feature"] == "TenureBand"]
    assert tenure["level"].tolist()[0] == "1-6" and tenure["customers"].sum() == len(X)
    assert set(table.loc[table["feature"] == "SeniorCitizen", "level"]) <= {"No", "Yes"}
