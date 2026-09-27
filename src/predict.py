"""Inference: load the saved pipeline and score customers.

    python -m src.predict                                   # Kaggle submission for data/raw/test.csv
    python -m src.predict --input customers.csv --output scored.csv

The saved pipeline contains all preprocessing, so raw customer rows go straight in.
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import load_config, resolve_path
from src.data import load_raw
from src.features import CATEGORY_LEVELS, INTERNET_ADDONS, NUMERIC_FEATURES, RAW_FEATURES

# Example customers for the demo app and tests.
EXAMPLE_CUSTOMERS = {
    "New fiber customer, month-to-month, electronic check": {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 2,
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "Yes", "StreamingMovies": "No", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.5, "TotalCharges": 171.0,
    },
    "Long-standing DSL customer, two-year contract": {
        "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes", "tenure": 60,
        "PhoneService": "Yes", "MultipleLines": "Yes", "InternetService": "DSL",
        "OnlineSecurity": "Yes", "OnlineBackup": "Yes", "DeviceProtection": "Yes", "TechSupport": "Yes",
        "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Two year",
        "PaperlessBilling": "No", "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 70.0, "TotalCharges": 4200.0,
    },
    "Phone-only customer, one-year contract": {
        "gender": "Female", "SeniorCitizen": 1, "Partner": "Yes", "Dependents": "No", "tenure": 24,
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "No",
        "OnlineSecurity": "No internet service", "OnlineBackup": "No internet service",
        "DeviceProtection": "No internet service", "TechSupport": "No internet service",
        "StreamingTV": "No internet service", "StreamingMovies": "No internet service",
        "Contract": "One year", "PaperlessBilling": "No", "PaymentMethod": "Mailed check",
        "MonthlyCharges": 20.0, "TotalCharges": 480.0,
    },
}


def load_model(model_path: Path | None = None, metadata_path: Path | None = None) -> tuple[Pipeline, dict]:
    """Load the fitted pipeline and its metadata (decision threshold, metrics, versions)."""
    cfg = load_config()
    model_path = model_path or resolve_path(cfg["paths"]["model"])
    metadata_path = metadata_path or resolve_path(cfg["paths"]["metadata"])
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)
    return joblib.load(model_path), metadata


def prepare_customer(customer: dict) -> pd.DataFrame:
    """Validate one customer's inputs and resolve contradictory service combinations.

    Raises ValueError for missing fields, unknown categories or negative numbers. A customer
    without internet cannot have internet add-ons, and one without phone service cannot have
    multiple lines, so those fields are set to the matching "No ... service" level.
    """
    missing = [col for col in RAW_FEATURES if col not in customer]
    if missing:
        raise ValueError(f"Missing fields: {missing}")
    row = {col: customer[col] for col in RAW_FEATURES}

    for col, levels in CATEGORY_LEVELS.items():
        if row[col] not in levels:
            raise ValueError(f"{col} must be one of {levels}, got {row[col]!r}")
    for col in NUMERIC_FEATURES:
        row[col] = float(row[col])
        if row[col] < 0:
            raise ValueError(f"{col} cannot be negative, got {row[col]}")
    if row["SeniorCitizen"] not in (0, 1):
        raise ValueError(f"SeniorCitizen must be 0 or 1, got {row['SeniorCitizen']}")

    for col in INTERNET_ADDONS:
        if row["InternetService"] == "No":
            row[col] = "No internet service"
        elif row[col] == "No internet service":
            row[col] = "No"
    if row["PhoneService"] == "No":
        row["MultipleLines"] = "No phone service"
    elif row["MultipleLines"] == "No phone service":
        row["MultipleLines"] = "No"
    return pd.DataFrame([row], columns=RAW_FEATURES)


def predict_customer(customer: dict, pipeline: Pipeline, threshold: float) -> dict:
    """Churn probability for one customer, plus the yes/no call at the model's decision threshold."""
    proba = float(pipeline.predict_proba(prepare_customer(customer))[0, 1])
    return {"churn_probability": proba, "will_churn": proba >= threshold, "threshold": threshold}


def score_file(input_path: Path, output_path: Path, pipeline: Pipeline, threshold: float,
               id_column: str = "id") -> pd.DataFrame:
    """Score every row of a raw CSV; keeps the id column when present."""
    df = load_raw(input_path)
    proba = pipeline.predict_proba(df[RAW_FEATURES])[:, 1]
    scored = pd.DataFrame({"churn_probability": proba, "will_churn": (proba >= threshold).astype(int)})
    if id_column in df:
        scored.insert(0, id_column, df[id_column].to_numpy())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)
    return scored


def write_submission(cfg: dict, pipeline: Pipeline) -> pd.DataFrame:
    """Kaggle submission: churn probability for every test id, in sample_submission order."""
    id_col, target = cfg["data"]["id_column"], cfg["data"]["target"]
    test = load_raw(resolve_path(cfg["paths"]["raw_test"]))
    template = pd.read_csv(resolve_path(cfg["paths"]["sample_submission"]), usecols=[id_col])
    scores = pd.DataFrame({id_col: test[id_col], target: pipeline.predict_proba(test[RAW_FEATURES])[:, 1]})
    submission = template.merge(scores, on=id_col, how="left", validate="one_to_one")
    if submission[target].isna().any():
        raise ValueError("Some sample_submission ids are missing from test.csv")
    path = resolve_path(cfg["paths"]["submission"])
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)
    return submission


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, help="raw CSV to score (default: Kaggle test set -> submission)")
    parser.add_argument("--output", type=Path, help="where to write the scored CSV (required with --input)")
    args = parser.parse_args(argv)

    cfg = load_config()
    pipeline, metadata = load_model()
    if args.input:
        if not args.output:
            parser.error("--output is required with --input")
        scored = score_file(args.input, args.output, pipeline, metadata["threshold"], cfg["data"]["id_column"])
        print(f"Scored {len(scored):,} customers -> {args.output} "
              f"({scored['will_churn'].mean():.1%} flagged at threshold {metadata['threshold']:.3f})")
    else:
        submission = write_submission(cfg, pipeline)
        print(f"Wrote {len(submission):,} predictions with {metadata['display_name']} -> "
              f"{resolve_path(cfg['paths']['submission'])}")


if __name__ == "__main__":
    main()
