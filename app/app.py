"""Gradio demo: enter a customer's details and get their churn probability.

Run locally with `python app/app.py`, then open http://127.0.0.1:7860.
The same file runs on Hugging Face Spaces (see scripts/deploy_space.py).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402

from src.features import CATEGORY_LEVELS, RAW_FEATURES  # noqa: E402
from src.predict import EXAMPLE_CUSTOMERS, load_model, predict_customer  # noqa: E402

PIPELINE, METADATA = load_model()
THRESHOLD = METADATA["threshold"]
REPO_URL = "https://github.com/aviraj1805/Customer-Churn-Prediction"

# Segment-level churn rates from the exploratory analysis (notebooks/02_eda.ipynb).
RISK_TRAITS = [
    (lambda c: c["Contract"] == "Month-to-month", "Month-to-month contract (42% churn vs 1% on two-year)"),
    (lambda c: c["PaymentMethod"] == "Electronic check", "Pays by electronic check (49% churn)"),
    (lambda c: c["InternetService"] == "Fiber optic", "Fiber optic internet (41.5% churn)"),
    (lambda c: c["InternetService"] != "No" and c["TechSupport"] == "No", "No tech support (40% churn)"),
    (lambda c: c["tenure"] <= 12, "In the first year with the company (about 50% churn)"),
]

LABELS = {
    "gender": "Gender", "SeniorCitizen": "Senior citizen", "Partner": "Has a partner",
    "Dependents": "Has dependents", "tenure": "Tenure (months)", "Contract": "Contract",
    "PaperlessBilling": "Paperless billing", "PaymentMethod": "Payment method",
    "MonthlyCharges": "Monthly charges ($)", "TotalCharges": "Total charges to date ($)",
    "PhoneService": "Phone service", "MultipleLines": "Multiple lines", "InternetService": "Internet service",
    "OnlineSecurity": "Online security", "OnlineBackup": "Online backup", "DeviceProtection": "Device protection",
    "TechSupport": "Tech support", "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming movies",
}
SECTIONS = {  # section -> rows of inputs
    "Account": [["tenure", "MonthlyCharges", "TotalCharges"], ["Contract", "PaymentMethod", "PaperlessBilling"]],
    "Services": [["PhoneService", "MultipleLines", "InternetService"],
                 ["OnlineSecurity", "OnlineBackup", "DeviceProtection"],
                 ["TechSupport", "StreamingTV", "StreamingMovies"]],
    "Demographics": [["gender", "SeniorCitizen", "Partner", "Dependents"]],
}
# The form opens pre-filled with the first example; total charges left empty to show the estimate.
DEFAULT_CUSTOMER = {**next(iter(EXAMPLE_CUSTOMERS.values())), "TotalCharges": None}


def make_input(feature: str):
    label, default = LABELS[feature], DEFAULT_CUSTOMER[feature]
    if feature == "tenure":
        return gr.Slider(1, 72, value=default, step=1, label=label)
    if feature == "MonthlyCharges":
        return gr.Slider(18, 120, value=default, step=0.5, label=label)
    if feature == "TotalCharges":
        return gr.Number(value=default, label=label, minimum=0,
                         info="Leave empty (or 0) to estimate as tenure x monthly charges")
    if feature == "SeniorCitizen":
        return gr.Radio(["No", "Yes"], value="Yes" if default else "No", label=label)
    levels = CATEGORY_LEVELS[feature]
    if len(levels) == 2:
        return gr.Radio(levels, value=default, label=label)
    return gr.Dropdown(levels, value=default, label=label)


def to_ui_values(customer: dict) -> list:
    """Customer dict -> values in RAW_FEATURES order, as the UI components expect them."""
    return [("Yes" if customer[f] else "No") if f == "SeniorCitizen" else customer[f] for f in RAW_FEATURES]


def predict(*values):
    customer = dict(zip(RAW_FEATURES, values))
    customer["SeniorCitizen"] = 1 if customer["SeniorCitizen"] == "Yes" else 0
    if not customer["TotalCharges"]:  # empty or 0: impossible for tenure >= 1, so estimate it
        customer["TotalCharges"] = customer["tenure"] * customer["MonthlyCharges"]
    try:
        result = predict_customer(customer, PIPELINE, THRESHOLD)
    except ValueError as err:
        raise gr.Error(str(err)) from err

    proba = result["churn_probability"]
    if result["will_churn"]:
        verdict = f"### Likely to churn\nChurn probability **{proba:.1%}** is above the decision threshold of {THRESHOLD:.1%}."
    else:
        verdict = f"### Unlikely to churn\nChurn probability **{proba:.1%}** is below the decision threshold of {THRESHOLD:.1%}."
    traits = [text for check, text in RISK_TRAITS if check(customer)]
    traits_md = "\n".join(f"- {t}" for t in traits) if traits else "- None of the main high-risk traits"
    details = (
        f"{verdict}\n\n**High-risk traits present** (from the data analysis):\n{traits_md}\n\n"
        f"<sub>The threshold was chosen to balance precision and recall (maximum F1) on training data. "
        f"Customers above it are the ones worth a retention offer.</sub>"
    )
    return {"Churn": proba, "No churn": 1 - proba}, details


metrics = METADATA["test_metrics"]
HEADER = f"""# Customer Churn Predictor
Estimate the probability that a telecom customer will leave, from their account, services and demographics.

**Model:** {METADATA["display_name"]} (scikit-learn pipeline) · **Hold-out ROC-AUC:** {metrics["roc_auc"]:.4f} ·
**Recall:** {metrics["recall"]:.1%} · **Precision:** {metrics["precision"]:.1%} ·
trained on {METADATA["n_train_rows"]:,} customers from the
[Kaggle Playground S6E3](https://www.kaggle.com/competitions/playground-series-s6e3) dataset ·
[Source code]({REPO_URL})
"""

with gr.Blocks(title="Customer Churn Predictor", theme=gr.themes.Soft()) as demo:
    gr.Markdown(HEADER)
    components = {}
    with gr.Row():
        with gr.Column(scale=3):
            for section, rows in SECTIONS.items():
                with gr.Group():
                    gr.Markdown(f"#### {section}")
                    for row in rows:
                        with gr.Row():
                            for feature in row:
                                components[feature] = make_input(feature)
            button = gr.Button("Predict churn", variant="primary")
        with gr.Column(scale=2):
            probability = gr.Label(label="Churn probability", num_top_classes=2)
            explanation = gr.Markdown()

    inputs = [components[f] for f in RAW_FEATURES]
    gr.Examples(
        examples=[to_ui_values(c) for c in EXAMPLE_CUSTOMERS.values()],
        example_labels=list(EXAMPLE_CUSTOMERS),
        inputs=inputs,
    )
    button.click(predict, inputs=inputs, outputs=[probability, explanation], api_name="predict")
    demo.load(predict, inputs=inputs, outputs=[probability, explanation], api_name=False)

if __name__ == "__main__":
    demo.launch()
