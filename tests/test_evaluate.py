import numpy as np
import pytest

from src.evaluate import choose_threshold, classification_metrics


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
