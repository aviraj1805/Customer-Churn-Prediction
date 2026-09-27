"""Evaluation: metrics, decision threshold, the model comparison table and report figures."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files; no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.models import DISPLAY_NAMES
from src.plot_style import BLUES, CATEGORICAL, GRID, HIGHLIGHT, INK, INK_SECONDARY, MUTED, SURFACE, apply_style

apply_style()
MODEL_COLORS = {  # color follows the model, never its rank; the untuned baseline recedes in gray
    "baseline": MUTED,
    **dict(zip(["logistic_regression", "random_forest", "hist_gradient_boosting", "xgboost", "lightgbm"],
               CATEGORICAL)),
}
CLASS_LABELS = ["No churn", "Churn"]


# --- Metrics ---------------------------------------------------------------------------------------
def choose_threshold(y_true, proba) -> float:
    """Probability cut-off that maximises F1 for the churn class.

    Called on out-of-fold predictions from the training data, never on the test set, so the
    hold-out metrics stay an honest estimate.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])  # the last PR point has no threshold


def classification_metrics(y_true, proba, threshold: float) -> dict:
    """Threshold-free ranking metrics plus accuracy / precision / recall / F1 at ``threshold``."""
    y_pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def comparison_table(results: dict) -> pd.DataFrame:
    """One row per model, sorted by hold-out ROC-AUC."""
    rows = [{
        "model": DISPLAY_NAMES[name],
        "cv_roc_auc": r["cv_auc_mean"],
        "cv_roc_auc_std": r["cv_auc_std"],
        **{f"test_{k}": v for k, v in r["test_metrics"].items()},
        "threshold": r["threshold"],
        "train_seconds": r["seconds"],
    } for name, r in results.items()]
    return pd.DataFrame(rows).sort_values("test_roc_auc", ascending=False).reset_index(drop=True)


def write_comparison(table: pd.DataFrame, results: dict, reports_dir: Path) -> None:
    """Save the comparison as CSV (for code) and Markdown (for the README)."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(reports_dir / "model_comparison.csv", index=False)

    lines = [
        "| Model | CV ROC-AUC | Test ROC-AUC | Test PR-AUC | Accuracy | Precision | Recall | F1 | Threshold |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in table.itertuples():
        lines.append(
            f"| {row.model} | {row.cv_roc_auc:.4f} ± {row.cv_roc_auc_std:.4f} | **{row.test_roc_auc:.4f}** | "
            f"{row.test_pr_auc:.4f} | {row.test_accuracy:.4f} | {row.test_precision:.4f} | "
            f"{row.test_recall:.4f} | {row.test_f1:.4f} | {row.threshold:.3f} |"
        )
    lines += ["", "## Best hyperparameters", ""]
    for name, r in results.items():
        params = ", ".join(f"`{k}={v}`" for k, v in r["best_params"].items())
        lines.append(f"- **{DISPLAY_NAMES[name]}**: {params}")
    (reports_dir / "model_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_grid_results(model_name: str, cv_results: dict, reports_dir: Path) -> None:
    """Save every grid-search candidate with its CV score, e.g. to show whether class weighting helped."""
    out_dir = reports_dir / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(cv_results["params"]).rename(columns=lambda c: c.removeprefix("model__"))
    table["cv_roc_auc_mean"] = cv_results["mean_test_score"]
    table["cv_roc_auc_std"] = cv_results["std_test_score"]
    table["rank"] = cv_results["rank_test_score"]
    table.sort_values("rank").to_csv(out_dir / f"{model_name}_grid.csv", index=False)


# --- Figures ---------------------------------------------------------------------------------------
def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _subtitle(ax, text: str) -> None:
    ax.text(0, 1.02, text, transform=ax.transAxes, color=INK_SECONDARY, fontsize=9, va="bottom")


def plot_roc_curves(y_true, probas: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1)
    for name, proba in sorted(probas.items(), key=lambda kv: -roc_auc_score(y_true, kv[1])):
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        ax.plot(fpr, tpr, color=MODEL_COLORS[name], linewidth=1.6, label=f"{DISPLAY_NAMES[name]}  (AUC {auc:.4f})")
    ax.set(xlabel="False positive rate", ylabel="True positive rate (recall)", xlim=(0, 1), ylim=(0, 1.01))
    ax.set_title("ROC curves", pad=22)
    _subtitle(ax, f"Hold-out test set, n = {len(y_true):,}")
    ax.legend(loc="lower right")
    _save(fig, path)


def plot_pr_curves(y_true, probas: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.axhline(np.mean(y_true), color=GRID, linewidth=1)
    for name, proba in sorted(probas.items(), key=lambda kv: -average_precision_score(y_true, kv[1])):
        precision, recall, _ = precision_recall_curve(y_true, proba)
        ap = average_precision_score(y_true, proba)
        ax.plot(recall, precision, color=MODEL_COLORS[name], linewidth=1.6,
                label=f"{DISPLAY_NAMES[name]}  (AP {ap:.4f})")
    ax.set(xlabel="Recall (share of churners caught)", ylabel="Precision (share of flags that churn)",
           xlim=(0, 1), ylim=(0, 1.01))
    ax.set_title("Precision-recall curves", pad=22)
    _subtitle(ax, f"Hold-out test set; flat line = churn rate ({np.mean(y_true):.1%})")
    ax.legend(loc="lower left", bbox_to_anchor=(0, np.mean(y_true) + 0.03))  # sit above the churn-rate line
    _save(fig, path)


def plot_model_comparison(table: pd.DataFrame, path: Path) -> None:
    """Dot plot of hold-out ROC-AUC; the winner is highlighted, the rest recede."""
    table = table.sort_values("test_roc_auc")
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(table) + 1.4))
    colors = [HIGHLIGHT if i == len(table) - 1 else MUTED for i in range(len(table))]
    ax.scatter(table["test_roc_auc"], table["model"], s=70, c=colors, edgecolors=SURFACE, linewidths=2, zorder=3)
    for row in table.itertuples():
        ax.annotate(f"{row.test_roc_auc:.4f}", (row.test_roc_auc, row.model), xytext=(9, 0),
                    textcoords="offset points", va="center", color=INK_SECONDARY, fontsize=9)
    span = table["test_roc_auc"].max() - table["test_roc_auc"].min()
    ax.set_xlim(table["test_roc_auc"].min() - 0.15 * span, table["test_roc_auc"].max() + 0.3 * span)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("ROC-AUC on hold-out test set (axis zoomed; higher is better)")
    ax.set_title("Model comparison", pad=22)
    _subtitle(ax, f"Hold-out test set; best and worst model are {span:.4f} ROC-AUC apart")
    _save(fig, path)


def _draw_confusion(ax, y_true, y_pred, title: str) -> None:
    counts = confusion_matrix(y_true, y_pred)
    shares = counts / counts.sum(axis=1, keepdims=True)  # row-normalised: share of each true class
    ax.imshow(shares, cmap=BLUES, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{counts[i, j]:,}\n{shares[i, j]:.1%}", ha="center", va="center", fontsize=9,
                    color="white" if shares[i, j] > 0.5 else INK)
    ax.set_xticks([0, 1], CLASS_LABELS)
    ax.set_yticks([0, 1], CLASS_LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.grid(False)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=10)


def plot_confusion_matrix(y_true, y_pred, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    _draw_confusion(ax, y_true, y_pred, title)
    _save(fig, path)


def plot_confusion_matrices(y_true, preds: dict, path: Path) -> None:
    """Small multiples: one confusion matrix per model, same color scale."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    for ax, (name, y_pred) in zip(axes.flat, preds.items()):
        _draw_confusion(ax, y_true, y_pred, DISPLAY_NAMES[name])
    for ax in list(axes.flat)[len(preds):]:
        ax.set_visible(False)
    fig.suptitle("Confusion matrices on the hold-out test set (each model at its own F1-optimal threshold)",
                 x=0.01, ha="left", fontsize=12, fontweight="semibold")
    fig.tight_layout()
    _save(fig, path)


def permutation_importances(pipeline, X, y, seed: int, n_rows: int = 20_000) -> pd.DataFrame:
    """Drop in ROC-AUC when each raw input column is shuffled (model-agnostic importance)."""
    sample = X.sample(n=min(n_rows, len(X)), random_state=seed)
    result = permutation_importance(pipeline, sample, y.loc[sample.index], scoring="roc_auc",
                                    n_repeats=5, random_state=seed)
    return (pd.DataFrame({"feature": X.columns, "importance": result.importances_mean,
                          "std": result.importances_std})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def plot_feature_importance(importances: pd.DataFrame, model_label: str, path: Path) -> None:
    data = importances.sort_values("importance")
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(data) + 1.4))
    ax.barh(data["feature"], data["importance"], height=0.55, color=HIGHLIGHT,
            xerr=data["std"], error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 0})
    for row in data.itertuples():
        ax.annotate(f"{row.importance:.4f}", (max(row.importance, 0) + row.std, row.feature), xytext=(5, 0),
                    textcoords="offset points", va="center", color=INK_SECONDARY, fontsize=8)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(right=data["importance"].max() * 1.18)
    ax.set_xlabel("Drop in ROC-AUC when the column is randomly shuffled")
    ax.set_title("What drives churn predictions", pad=22)
    _subtitle(ax, f"Permutation importance, {model_label}, hold-out sample of 20,000 customers")
    _save(fig, path)
