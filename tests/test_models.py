import numpy as np
import pytest

from src.config import load_config
from src.data import split_X_y
from src.models import baseline_pipeline, candidate_pipeline

MODEL_NAMES = list(load_config()["models"])


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_every_grid_parameter_is_valid(model_name, cfg):
    pipeline, grid = candidate_pipeline(model_name, cfg)
    for param, values in grid.items():
        pipeline.set_params(**{param: values[0]})  # raises ValueError on a typo in config.yaml


@pytest.mark.parametrize("model_name", MODEL_NAMES + ["baseline"])
def test_pipeline_trains_and_returns_probabilities(model_name, raw_df, cfg):
    X, y = split_X_y(raw_df, cfg)
    pipeline = baseline_pipeline(cfg) if model_name == "baseline" else candidate_pipeline(model_name, cfg)[0]
    proba = pipeline.fit(X, y).predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert np.all((proba >= 0) & (proba <= 1))
