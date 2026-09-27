"""Everything the app loads once at startup: the model, its metadata and the small report tables.

The report tables are written by `python -m src.train`, so the app never needs the raw data.
A missing table only hides the matching section; the prediction features keep working.
"""
from dataclasses import dataclass

import pandas as pd

from src.config import load_config, resolve_path
from src.predict import load_model


@dataclass(frozen=True)
class Artifacts:
    pipeline: object
    metadata: dict
    comparison: pd.DataFrame | None   # one row per model: CV and hold-out metrics
    thresholds: pd.DataFrame | None   # confusion counts and metrics at every cut-off (best model)
    curves: pd.DataFrame | None       # ROC and PR curve points per model
    importance: pd.DataFrame | None   # permutation importance per input column
    segments: pd.DataFrame | None     # churn rate per segment level

    @property
    def churn_rate(self) -> float:
        return self.metadata.get("churn_rate", 0.225)


def _read_report(name: str) -> pd.DataFrame | None:
    path = resolve_path(load_config()["paths"]["reports_dir"]) / name
    return pd.read_csv(path) if path.exists() else None


def load_artifacts() -> Artifacts:
    pipeline, metadata = load_model()
    return Artifacts(
        pipeline=pipeline,
        metadata=metadata,
        comparison=_read_report("model_comparison.csv"),
        thresholds=_read_report("threshold_analysis.csv"),
        curves=_read_report("curves.csv"),
        importance=_read_report("feature_importance.csv"),
        segments=_read_report("segment_churn_rates.csv"),
    )
