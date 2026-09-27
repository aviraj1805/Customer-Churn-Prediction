"""Per-customer explanations and what-if analysis for the churn model.

- ``contributions``: the model's built-in exact SHAP values (TreeSHAP) per model-ready column, in log-odds.
  Contributions plus the bias add up exactly to the model's log-odds for that customer.
- ``drivers``: those contributions summed back to the original input fields (e.g. all one-hot columns of
  Contract become one "Contract" contribution).
- ``retention_scenarios``: the model's prediction if one retention action were applied to the customer.
- ``tenure_outlook``: predicted risk for the same customer at every tenure.
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.features import CATEGORICAL_FEATURES, add_engineered_features
from src.predict import prepare_customer
from src.preprocessing import coerce_numeric, feature_names

# (label, applies to this customer?, changes). Charges are kept fixed, as if the change were offered for free.
RETENTION_ACTIONS = [
    ("Move to a one-year contract",
     lambda c: c["Contract"] == "Month-to-month", {"Contract": "One year"}),
    ("Move to a two-year contract",
     lambda c: c["Contract"] != "Two year", {"Contract": "Two year"}),
    ("Switch to automatic card payment",
     lambda c: c["PaymentMethod"] in ("Electronic check", "Mailed check"), {"PaymentMethod": "Credit card (automatic)"}),
    ("Add tech support",
     lambda c: c["InternetService"] != "No" and c["TechSupport"] == "No", {"TechSupport": "Yes"}),
    ("Add online security",
     lambda c: c["InternetService"] != "No" and c["OnlineSecurity"] == "No", {"OnlineSecurity": "Yes"}),
]


def contributions(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame | None:
    """SHAP contributions (log-odds) per model-ready column plus 'bias'; None if the model has no native support."""
    preprocess, model = pipeline.named_steps["preprocess"], pipeline.named_steps["model"]
    X_ready = preprocess.transform(X)
    kind = type(model).__name__
    if kind == "XGBClassifier":
        import xgboost

        values = model.get_booster().predict(xgboost.DMatrix(X_ready), pred_contribs=True)
    elif kind == "LGBMClassifier":
        values = model.predict(X_ready, pred_contrib=True)
    else:
        return None
    return pd.DataFrame(values, columns=[*feature_names(preprocess), "bias"])


def original_field(column: str) -> str:
    """Map a model-ready column back to its input field: 'categorical__Contract_Two year' -> 'Contract'."""
    kind, name = column.split("__", 1)
    if kind == "categorical":
        return next(field for field in CATEGORICAL_FEATURES if name.startswith(f"{field}_"))
    return name


def drivers(pipeline: Pipeline, customer: dict) -> pd.DataFrame | None:
    """How much each field pushed this customer's churn risk up (+) or down (-), largest effect first."""
    X = prepare_customer(customer)
    contrib = contributions(pipeline, X)
    if contrib is None:
        return None
    per_field = contrib.iloc[0].drop("bias").groupby(original_field).sum()
    values = add_engineered_features(coerce_numeric(X)).iloc[0]
    table = pd.DataFrame({
        "field": per_field.index,
        "value": [values.get(field) for field in per_field.index],
        "contribution": per_field.to_numpy(),
    })
    return table.reindex(table["contribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def retention_scenarios(pipeline: Pipeline, customer: dict) -> pd.DataFrame:
    """Predicted churn probability after each applicable retention action, biggest reduction first."""
    columns = ["action", "churn_probability", "change"]
    base = float(pipeline.predict_proba(prepare_customer(customer))[0, 1])
    actions = [(label, changes) for label, applies, changes in RETENTION_ACTIONS if applies(customer)]
    if not actions:
        return pd.DataFrame(columns=columns)
    X = pd.concat([prepare_customer({**customer, **changes}) for _, changes in actions], ignore_index=True)
    proba = pipeline.predict_proba(X)[:, 1]
    table = pd.DataFrame({"action": [label for label, _ in actions], "churn_probability": proba,
                          "change": proba - base})
    return table.sort_values("change").reset_index(drop=True)[columns]


def tenure_outlook(pipeline: Pipeline, customer: dict, max_tenure: int = 72) -> pd.DataFrame:
    """Predicted churn probability at every tenure, keeping the monthly bill and all services fixed."""
    tenures = np.arange(1, max_tenure + 1)
    X = prepare_customer(customer)
    X = X.loc[X.index.repeat(len(tenures))].reset_index(drop=True)
    X["tenure"] = tenures
    X["TotalCharges"] = tenures * X["MonthlyCharges"]
    return pd.DataFrame({"tenure": tenures, "churn_probability": pipeline.predict_proba(X)[:, 1]})
