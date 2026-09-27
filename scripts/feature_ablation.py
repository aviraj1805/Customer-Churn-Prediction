"""Reproduce the notebook's baseline through the new pipeline and measure the engineered features.

Uses the exact 5 folds of the original notebook (StratifiedKFold, shuffle, seed 42), so fold scores
can be compared one-to-one. Run: python -m scripts.feature_ablation  (takes ~4 minutes)
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import load_config
from src.data import load_dataset
from src.preprocessing import build_preprocessor

NOTEBOOK_XGBOOST_FOLDS = [0.91539, 0.91648, 0.91581, 0.91689, 0.91415]

MODELS = {
    "XGBoost (notebook params)": lambda seed: XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6, eval_metric="auc", random_state=seed, n_jobs=-1
    ),
    "Logistic Regression": lambda seed: LogisticRegression(max_iter=2000),
}


def main() -> None:
    cfg = load_config()
    X, y = load_dataset(cfg)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg["seed"])

    print(f"Original notebook XGBoost: mean AUC {np.mean(NOTEBOOK_XGBOOST_FOLDS):.5f}")
    for name, make_model in MODELS.items():
        for engineered in (False, True):
            pipeline = Pipeline([
                ("preprocess", build_preprocessor(engineered)),
                ("model", make_model(cfg["seed"])),
            ])
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
            print(f"{name:26s} engineered={engineered!s:5s} mean AUC {scores.mean():.5f} "
                  f"(std {scores.std():.5f}) folds {np.round(scores, 5).tolist()}")


if __name__ == "__main__":
    main()
