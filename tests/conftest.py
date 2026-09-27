"""Shared fixtures. Tests use synthetic rows with the real schema, so they run without the Kaggle data."""
import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.features import CATEGORY_LEVELS, INTERNET_ADDONS


def make_raw_frame(n_rows: int = 300, seed: int = 0, with_target: bool = True) -> pd.DataFrame:
    """Build a raw Kaggle-style frame with consistent service columns and a learnable churn signal."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({col: rng.choice(levels, n_rows) for col, levels in CATEGORY_LEVELS.items()})

    no_internet = df["InternetService"] == "No"
    for col in INTERNET_ADDONS:
        df.loc[no_internet, col] = "No internet service"
        df.loc[~no_internet & (df[col] == "No internet service"), col] = "No"
    no_phone = df["PhoneService"] == "No"
    df.loc[no_phone, "MultipleLines"] = "No phone service"
    df.loc[~no_phone & (df["MultipleLines"] == "No phone service"), "MultipleLines"] = "No"

    df["SeniorCitizen"] = rng.integers(0, 2, n_rows)
    df["tenure"] = rng.integers(1, 73, n_rows)
    df["MonthlyCharges"] = rng.uniform(18, 119, n_rows).round(2)
    df["TotalCharges"] = (df["tenure"] * df["MonthlyCharges"]).round(2)
    df.insert(0, "id", range(n_rows))

    if with_target:
        risk = 0.05 + 0.45 * (df["Contract"] == "Month-to-month") + 0.3 * (df["tenure"] < 12)
        df["Churn"] = np.where(rng.random(n_rows) < risk, "Yes", "No")
    return df


@pytest.fixture
def cfg() -> dict:
    return load_config()


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_raw_frame()


@pytest.fixture
def raw_csv(tmp_path, raw_df):
    path = tmp_path / "train.csv"
    raw_df.to_csv(path, index=False)
    return path
