"""Model registry: the candidate classifiers and how to build full pipelines from config.yaml."""
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.preprocessing import build_preprocessor

MODEL_CLASSES = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
    "xgboost": XGBClassifier,
    "lightgbm": LGBMClassifier,
}

DISPLAY_NAMES = {
    "baseline": "XGBoost (original notebook)",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "HistGradientBoosting",
    "xgboost": "XGBoost (tuned)",
    "lightgbm": "LightGBM (tuned)",
}


def build_pipeline(model_name: str, params: dict, cfg: dict) -> Pipeline:
    """Preprocessing + classifier in one Pipeline, so both are fitted and saved together."""
    model = MODEL_CLASSES[model_name](**params)
    if "random_state" in model.get_params():
        model.set_params(random_state=cfg["seed"])
    return Pipeline([
        ("preprocess", build_preprocessor(cfg["features"]["engineered"])),
        ("model", model),
    ])


def candidate_pipeline(model_name: str, cfg: dict) -> tuple[Pipeline, dict]:
    """Pipeline with the model's fixed params, and its search grid keyed for GridSearchCV.

    Grid keys get the ``model__`` prefix so GridSearchCV tunes the classifier step of the pipeline.
    """
    spec = cfg["models"][model_name]
    pipeline = build_pipeline(model_name, spec.get("params", {}), cfg)
    grid = {f"model__{param}": values for param, values in spec["grid"].items()}
    return pipeline, grid


def baseline_pipeline(cfg: dict) -> Pipeline:
    """The original notebook's model, untuned, for the reference row of the results table."""
    return build_pipeline(cfg["baseline"]["model"], cfg["baseline"]["params"], cfg)
