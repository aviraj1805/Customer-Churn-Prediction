"""Feature schema (which raw columns the model uses) and engineered features."""
import pandas as pd

NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

# Allowed values per categorical column, as they appear in the Kaggle data.
CATEGORY_LEVELS = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}
CATEGORICAL_FEATURES = list(CATEGORY_LEVELS)

# Add-on services that only exist for customers with an internet connection.
INTERNET_ADDONS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# The 19 raw input columns, in the order the model expects them.
RAW_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

ENGINEERED_FEATURES = ["AvgChargePerMonth", "ChargeIncrease", "NumAddonServices"]


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add domain features that tree splits and linear models cannot build on their own.

    - AvgChargePerMonth: lifetime spend divided by tenure (what the customer paid on average)
    - ChargeIncrease: current monthly bill minus that average (a recent price rise)
    - NumAddonServices: how many internet add-ons the customer subscribes to
    """
    X = X.copy()
    X["AvgChargePerMonth"] = X["TotalCharges"] / X["tenure"].clip(lower=1)
    X["ChargeIncrease"] = X["MonthlyCharges"] - X["AvgChargePerMonth"]
    X["NumAddonServices"] = sum((X[col] == "Yes").astype(int) for col in INTERNET_ADDONS)
    return X
