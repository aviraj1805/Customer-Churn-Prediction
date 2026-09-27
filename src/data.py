"""Data loading and splitting.

Everything that touches the raw CSV files lives here, so the rest of the code works with
a feature frame ``X`` (the 19 raw columns) and a 0/1 target ``y``.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import load_config, resolve_path
from src.features import CATEGORICAL_FEATURES, RAW_FEATURES


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read a raw Kaggle CSV and check that every expected feature column is present.

    Categorical columns are read as pandas ``category`` dtype, which cuts memory use
    roughly tenfold on the 594k-row training file.
    """
    df = pd.read_csv(path, dtype={col: "category" for col in CATEGORICAL_FEATURES})
    missing = sorted(set(RAW_FEATURES) - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")
    return df


def encode_target(y: pd.Series, positive_label: str = "Yes") -> pd.Series:
    """Map the Yes/No churn label to 1/0."""
    if y.isna().any():
        raise ValueError("Target column contains missing values")
    return (y == positive_label).astype(int)


def split_X_y(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Select the model's input columns (dropping ``id``) and encode the target."""
    X = df[RAW_FEATURES]
    y = encode_target(df[cfg["data"]["target"]], cfg["data"]["positive_label"])
    return X, y


def load_dataset(cfg: dict | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Load the labelled Kaggle training file as ``(X, y)``."""
    cfg = cfg or load_config()
    df = load_raw(resolve_path(cfg["paths"]["raw_train"]))
    return split_X_y(df, cfg)


def train_test(X: pd.DataFrame, y: pd.Series, cfg: dict):
    """Stratified hold-out split, so both parts keep the 22.5% churn rate."""
    return train_test_split(
        X, y, test_size=cfg["data"]["test_size"], stratify=y, random_state=cfg["seed"]
    )


def stratified_sample(
    X: pd.DataFrame, y: pd.Series, n_rows: int | None, seed: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Draw ``n_rows`` rows with the same class balance as ``y``; ``None`` keeps everything."""
    if n_rows is None or n_rows >= len(X):
        return X, y
    X_sample, _, y_sample, _ = train_test_split(
        X, y, train_size=n_rows, stratify=y, random_state=seed
    )
    return X_sample, y_sample
