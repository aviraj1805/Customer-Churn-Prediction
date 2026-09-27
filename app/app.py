"""Customer Churn Predictor: Gradio web app.

Run locally with `python app/app.py` and open http://127.0.0.1:7860. Deployed on Render (see render.yaml).
"""
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # no usage telemetry from the app

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402

from app import components as ui  # noqa: E402
from app.artifacts import load_artifacts  # noqa: E402
from app.theme import CSS, THEME  # noqa: E402
from src.data import load_raw  # noqa: E402
from src.explain import drivers, retention_scenarios, tenure_outlook  # noqa: E402
from src.features import CATEGORY_LEVELS, INTERNET_ADDONS, RAW_FEATURES  # noqa: E402
from src.predict import (  # noqa: E402
    EXAMPLE_CUSTOMERS,
    RISK_TIERS,
    predict_customer,
    score_frame,
    unknown_categories,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("churn-app")

started = time.perf_counter()
ART = load_artifacts()
THRESHOLD = ART.metadata["threshold"]
log.info("Loaded %s (threshold %.3f) in %.1fs", ART.metadata["display_name"], THRESHOLD, time.perf_counter() - started)

DEFAULT_CUSTOMER = {**next(iter(EXAMPLE_CUSTOMERS.values())), "TotalCharges": 0}
FORM_SECTIONS = {  # section -> rows of inputs
    "Account": [["tenure", "MonthlyCharges", "TotalCharges"], ["Contract", "PaymentMethod", "PaperlessBilling"]],
    "Services": [["PhoneService", "MultipleLines", "InternetService"],
                 ["OnlineSecurity", "OnlineBackup", "DeviceProtection"],
                 ["TechSupport", "StreamingTV", "StreamingMovies"]],
    "Demographics": [["gender", "SeniorCitizen", "Partner", "Dependents"]],
}
OUTLOOK_COLORS = {"This customer": "#4f46e5", "Decision threshold": "#898781"}


# --- Form helpers ------------------------------------------------------------------------------------
def dependent_choices(field: str, parent_value: str) -> list[str]:
    """Add-ons need internet, multiple lines need phone service: offer only the options that make sense."""
    if field in INTERNET_ADDONS:
        return ["No internet service"] if parent_value == "No" else ["No", "Yes"]
    if field == "MultipleLines":
        return ["No phone service"] if parent_value == "No" else ["No", "Yes"]
    return CATEGORY_LEVELS[field]


def dependent_update(field: str, parent_value: str, current: str):
    choices = dependent_choices(field, parent_value)
    return gr.update(choices=choices, value=current if current in choices else choices[0],
                     interactive=len(choices) > 1)


def make_input(field: str, customer: dict):
    label, value = ui.FIELD_LABELS[field], customer[field]
    if field == "tenure":
        return gr.Slider(1, 72, value=value, step=1, label=label, info="Months with the company")
    if field == "MonthlyCharges":
        return gr.Slider(18, 120, value=value, step=0.5, label=label, info="Current monthly bill")
    if field == "TotalCharges":
        return gr.Number(value=value, minimum=0, label=label, info="Leave at 0 to estimate as tenure x monthly bill")
    if field == "SeniorCitizen":
        return gr.Radio(["No", "Yes"], value="Yes" if value else "No", label=label)
    if field == "PaymentMethod":
        return gr.Dropdown(CATEGORY_LEVELS[field], value=value, label=label)
    parent = {"MultipleLines": "PhoneService", **{f: "InternetService" for f in INTERNET_ADDONS}}.get(field)
    choices = dependent_choices(field, customer[parent]) if parent else CATEGORY_LEVELS[field]
    return gr.Radio(choices, value=value, label=label, interactive=len(choices) > 1)


def to_customer(values) -> tuple[dict, bool]:
    """UI values (in RAW_FEATURES order) -> model input; also says whether TotalCharges was estimated."""
    customer = dict(zip(RAW_FEATURES, values))
    customer["SeniorCitizen"] = 1 if customer["SeniorCitizen"] == "Yes" else 0
    estimated = not customer["TotalCharges"]  # empty or 0 is impossible for tenure >= 1
    if estimated:
        customer["TotalCharges"] = customer["tenure"] * customer["MonthlyCharges"]
    return customer, estimated


def to_ui_values(customer: dict) -> list:
    return [("Yes" if customer[f] else "No") if f == "SeniorCitizen" else customer[f] for f in RAW_FEATURES]


# --- Predict tab ---------------------------------------------------------------------------------------
def analyse(*values):
    """Everything the Predict tab shows for one customer: gauge, verdict, drivers, what-ifs, tenure outlook."""
    customer, estimated = to_customer(values)
    try:
        result = predict_customer(customer, ART.pipeline, THRESHOLD)
    except ValueError as err:
        raise gr.Error(str(err)) from err
    proba = result["churn_probability"]

    outlook = tenure_outlook(ART.pipeline, customer)
    plot_data = pd.concat([
        pd.DataFrame({"tenure": outlook["tenure"], "churn_probability": outlook["churn_probability"] * 100,
                      "series": "This customer"}),
        pd.DataFrame({"tenure": [1, 72], "churn_probability": [THRESHOLD * 100] * 2, "series": "Decision threshold"}),
    ], ignore_index=True)
    later = min(int(customer["tenure"]) + 12, 72)
    later_proba = outlook.loc[outlook["tenure"] == later, "churn_probability"].iloc[0]
    outlook_note = (f'<div class="caption">Same plan and bill, only tenure changes. If this customer stays until month '
                    f'{later}, predicted risk goes from <b>{ui.pct(proba)}</b> to <b>{ui.pct(later_proba)}</b>.</div>')
    if estimated:
        outlook_note += (f'<div class="caption">Total charges estimated as '
                         f'${customer["TotalCharges"]:,.2f} (tenure x monthly bill).</div>')

    return (
        ui.gauge_html(proba, THRESHOLD),
        ui.verdict_html(proba, THRESHOLD, ART.churn_rate),
        ui.drivers_html(drivers(ART.pipeline, customer)),
        ui.scenarios_html(retention_scenarios(ART.pipeline, customer), proba),
        plot_data,
        outlook_note,
    )


def build_predict_tab():
    inputs = {}
    with gr.Row(elem_classes="preset-row"):
        preset_buttons = {name: gr.Button(name, size="sm", variant="secondary") for name in EXAMPLE_CUSTOMERS}
        reset_button = gr.Button("Reset", size="sm")
    with gr.Row(equal_height=False):
        with gr.Column(scale=7):
            for section, rows in FORM_SECTIONS.items():
                with gr.Group():
                    gr.Markdown(f"**{section}**", elem_classes="section-title", padding=True)
                    for row in rows:
                        with gr.Row():
                            for field in row:
                                inputs[field] = make_input(field, DEFAULT_CUSTOMER)
        with gr.Column(scale=5, elem_classes="sticky-col"):
            with gr.Group():
                gauge = gr.HTML(padding=True)
                verdict = gr.HTML(padding=True)
            gr.Markdown("### Why this prediction")
            driver_bars = gr.HTML()
            gr.HTML('<div class="caption">Each bar is that input\'s contribution to this customer\'s risk score '
                    '(SHAP values from the model, in log-odds). Hover a bar for details.</div>')
    with gr.Row(equal_height=False):
        with gr.Column():
            gr.Markdown("### What would reduce the risk?")
            scenarios = gr.HTML()
            gr.HTML('<div class="caption">The model re-scores the customer with one change at a time, keeping the '
                    'monthly bill the same (as if the change were offered for free).</div>')
        with gr.Column():
            gr.Markdown("### Risk outlook by tenure")
            outlook_plot = gr.LinePlot(
                x="tenure", y="churn_probability", color="series", color_map=OUTLOOK_COLORS,
                x_title="Tenure (months)", y_title="Churn probability (%)", x_lim=[1, 72], y_lim=[0, 100], height=260,
                tooltip=["tenure", "churn_probability", "series"], show_label=False,
            )
            outlook_note = gr.HTML()

    ordered = [inputs[f] for f in RAW_FEATURES]
    outputs = [gauge, verdict, driver_bars, scenarios, outlook_plot, outlook_note]

    # Keep dependent options consistent (no add-ons without internet, no extra lines without phone).
    addons = [inputs[f] for f in INTERNET_ADDONS]
    inputs["InternetService"].change(
        lambda internet, *current: [dependent_update(f, internet, c) for f, c in zip(INTERNET_ADDONS, current)],
        [inputs["InternetService"], *addons], addons, api_name=False, show_progress="hidden")
    inputs["PhoneService"].change(
        lambda phone, current: dependent_update("MultipleLines", phone, current),
        [inputs["PhoneService"], inputs["MultipleLines"]], inputs["MultipleLines"], api_name=False,
        show_progress="hidden")

    for name, button in preset_buttons.items():
        button.click(lambda c=EXAMPLE_CUSTOMERS[name]: to_ui_values(c), None, ordered, api_name=False)
    reset_button.click(lambda: to_ui_values(DEFAULT_CUSTOMER), None, ordered, api_name=False)

    gr.on([c.change for c in ordered], analyse, ordered, outputs, trigger_mode="always_last",
          show_progress="hidden", api_name=False)
    return ordered, outputs


# --- Batch scoring tab ---------------------------------------------------------------------------------
MAX_BATCH_ROWS = 50_000  # keeps memory safe on the free 512 MB instance
SAMPLE_CSV = ROOT / "app" / "assets" / "sample_customers.csv"
TIER_HEX = {"Low": "#0ca30c", "Moderate": "#fab219", "High": "#ec835a", "Very high": "#d03b3b"}
CONTEXT_COLUMNS = ["Contract", "tenure", "MonthlyCharges", "PaymentMethod", "InternetService"]
TOP_TABLE_NAMES = {"id": "ID", "churn_probability": "Churn probability (%)", "risk_tier": "Risk tier",
                   "Contract": "Contract", "tenure": "Tenure", "MonthlyCharges": "Monthly ($)"}


def score_batch(file_path):
    """Score an uploaded CSV: summary cards, tier distribution, top-risk table and a download."""
    if not file_path:
        return "", None, None, gr.update(visible=False)
    started = time.perf_counter()
    try:
        df = load_raw(file_path)
    except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as err:
        raise gr.Error(f"Could not read this file: {err}") from err
    if df.empty:
        raise gr.Error("The file has no customer rows.")
    if len(df) > MAX_BATCH_ROWS:
        raise gr.Error(f"Please upload at most {MAX_BATCH_ROWS:,} rows at a time (this file has {len(df):,}).")
    unknown = unknown_categories(df)
    if unknown:
        gr.Warning("Some values were not seen in training and are scored as unknown: "
                   + ", ".join(f"{col} ({n} rows)" for col, n in unknown.items()))

    scored = pd.concat([score_frame(df, ART.pipeline, THRESHOLD), df[CONTEXT_COLUMNS].reset_index(drop=True)], axis=1)
    out_path = Path(tempfile.mkdtemp()) / "churn_scores.csv"
    scored.to_csv(out_path, index=False)
    seconds = time.perf_counter() - started
    log.info("Scored batch of %d customers in %.2fs", len(scored), seconds)

    flagged = int(scored["will_churn"].sum())
    summary = ui.kpi_cards_html([
        ("Customers scored", f"{len(scored):,}", f"in {seconds * 1000:.0f} ms" if seconds < 1 else f"in {seconds:.1f} s"),
        ("Flagged for retention", f"{flagged:,}", f"{flagged / len(scored):.1%} at the {ui.pct(THRESHOLD)} threshold"),
        ("Expected churners", f"{scored['churn_probability'].sum():,.0f}", "sum of predicted probabilities"),
        ("Average risk", ui.pct(scored["churn_probability"].mean()), f"vs {ui.pct(ART.churn_rate)} in training data"),
    ])
    tiers = (scored["risk_tier"].value_counts().reindex(RISK_TIERS, fill_value=0)
             .rename_axis("risk_tier").reset_index(name="customers"))
    top = scored.sort_values("churn_probability", ascending=False).head(25)
    top = (top.assign(churn_probability=(top["churn_probability"].astype(float) * 100).round(1))
           [[c for c in TOP_TABLE_NAMES if c in top]].rename(columns=TOP_TABLE_NAMES))
    return summary, tiers, top, gr.update(value=str(out_path), visible=True)


def build_batch_tab():
    gr.Markdown("Score a whole customer list at once. Upload a CSV with the 19 input columns, in the same format "
                "as the Kaggle data. An `id` column is kept if present. Up to 50,000 rows per file.")
    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            upload = gr.File(label="Customer CSV", file_types=[".csv"], type="filepath", height=150)
        with gr.Column(scale=2):
            sample_button = gr.Button("Try it with 40 sample customers", variant="primary")
            gr.DownloadButton("Download the sample CSV", value=str(SAMPLE_CSV), variant="secondary")
            gr.HTML('<div class="caption">Columns: gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, '
                    'MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, '
                    'StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, '
                    'TotalCharges. See the sample file for valid values.</div>')
    summary = gr.HTML()
    with gr.Row(equal_height=False):
        with gr.Column(scale=2):
            tier_plot = gr.BarPlot(
                x="risk_tier", y="customers", color="risk_tier", color_map=TIER_HEX, sort=RISK_TIERS,
                x_title="Risk tier", y_title="Customers", height=300, show_label=False,
                tooltip=["risk_tier", "customers"],
            )
        with gr.Column(scale=3):
            top_table = gr.Dataframe(label="Highest-risk customers (top 25)", interactive=False, wrap=False)
    download = gr.DownloadButton("Download all scores (CSV)", visible=False, variant="primary")

    outputs = [summary, tier_plot, top_table, download]
    upload.upload(score_batch, upload, outputs, api_name=False)
    upload.clear(lambda: score_batch(None), None, outputs, api_name=False)
    sample_button.click(lambda: score_batch(str(SAMPLE_CSV)), None, outputs, api_name=False)


# --- Model performance tab -----------------------------------------------------------------------------
METRIC_COLORS = {"Precision": "#2a78d6", "Recall": "#eb6834", "F1": "#1baf7a"}
CURVE_AXES = {"roc": ("False positive rate", "True positive rate (recall)"), "pr": ("Recall", "Precision")}


def threshold_view(threshold: float):
    """Precision/recall cards, confusion matrix and campaign view at the chosen cut-off (hold-out data)."""
    table = ART.thresholds
    row = table.iloc[(table["threshold"] - threshold).abs().argmin()]
    cards = ui.kpi_cards_html([
        ("Precision", ui.pct(row["precision"]), "of flagged customers churn"),
        ("Recall", ui.pct(row["recall"]), "of churners are caught"),
        ("F1 score", f"{row['f1']:.3f}", "balance of the two"),
        ("Flagged", ui.pct(row["flagged_rate"]), "of all customers"),
    ])
    return cards, ui.confusion_html(row), ui.business_html(row)


def tradeoff_data() -> pd.DataFrame:
    flagged_any = ART.thresholds[ART.thresholds["flagged_rate"] > 0]  # precision is undefined if nobody is flagged
    long = flagged_any.melt(id_vars="threshold", value_vars=["precision", "recall", "f1"],
                               var_name="metric", value_name="value")
    long["metric"] = long["metric"].map({"precision": "Precision", "recall": "Recall", "f1": "F1"})
    long["value"] *= 100
    return long


def comparison_display() -> pd.DataFrame:
    table = ART.comparison
    deployed = ART.metadata["display_name"]
    return pd.DataFrame({
        "Model": [f"{m}  (deployed)" if m == deployed else m for m in table["model"]],
        "CV ROC-AUC": [f"{m:.4f} ± {s:.4f}" for m, s in zip(table["cv_roc_auc"], table["cv_roc_auc_std"])],
        "Test ROC-AUC": table["test_roc_auc"].map("{:.4f}".format),
        "Test PR-AUC": table["test_pr_auc"].map("{:.4f}".format),
        "Accuracy": table["test_accuracy"].map("{:.1%}".format),
        "Precision": table["test_precision"].map("{:.1%}".format),
        "Recall": table["test_recall"].map("{:.1%}".format),
        "F1": table["test_f1"].map("{:.3f}".format),
        "Threshold": table["threshold"].map("{:.3f}".format),
    })


def curve_data(curve: str) -> pd.DataFrame:
    points = ART.curves[ART.curves["curve"] == curve][["model", "x", "y"]]
    if curve == "roc":
        chance = pd.DataFrame({"model": "Random guess", "x": [0.0, 1.0], "y": [0.0, 1.0]})
    else:
        chance = pd.DataFrame({"model": "Random guess", "x": [0.0, 1.0], "y": [ART.churn_rate] * 2})
    return pd.concat([points, chance], ignore_index=True)


def model_details_md() -> str:
    md = ART.metadata
    params = ", ".join(f"`{k}={v}`" for k, v in md["best_params"].items())
    versions = ", ".join(f"{k} {v}" for k, v in md["library_versions"].items())
    return (
        f"**{md['display_name']}** inside a scikit-learn pipeline (imputation, scaling, one-hot encoding, "
        f"3 engineered features)\n\n"
        f"- Hyperparameters: {params}\n"
        f"- Decision threshold: **{md['threshold']:.3f}** (maximises F1 on out-of-fold training predictions)\n"
        f"- Cross-validated ROC-AUC: **{md['cv_roc_auc_mean']:.4f} ± {md['cv_roc_auc_std']:.4f}** (5 folds)\n"
        f"- Trained on {md['n_train_rows']:,} customers, tested on {md['n_test_rows']:,} held-out customers\n"
        f"- Trained {md['trained_at'][:10]} with {versions}"
    )


def build_performance_tab():
    m = ART.metadata["test_metrics"]
    gr.HTML(ui.kpi_cards_html([
        ("ROC-AUC", f"{m['roc_auc']:.4f}", "ranking quality (1 = perfect)"),
        ("PR-AUC", f"{m['pr_auc']:.4f}", f"vs {ART.churn_rate:.3f} for random"),
        ("Accuracy", ui.pct(m["accuracy"]), "at the decision threshold"),
        ("Precision", ui.pct(m["precision"]), "of flagged customers churn"),
        ("Recall", ui.pct(m["recall"]), "of churners are caught"),
        ("F1 score", f"{m['f1']:.3f}", "balance of the two"),
    ]))
    gr.HTML(f'<div class="caption">All numbers are measured on {ART.metadata["n_test_rows"]:,} customers that were '
            'held out from training, tuning and threshold selection.</div>')

    if ART.thresholds is not None:
        gr.Markdown("### Explore the decision threshold")
        gr.HTML('<div class="caption">A lower threshold catches more churners but sends more offers to customers '
                'who would have stayed. Move the slider to see the trade-off on the hold-out customers.</div>')
        initial = threshold_view(THRESHOLD)
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                slider = gr.Slider(0.05, 0.95, value=round(THRESHOLD, 2), step=0.01,
                                   label="Flag customers whose churn probability is at least")
                reset = gr.Button(f"Back to the recommended threshold ({THRESHOLD:.2f}, best F1)", size="sm")
                cards = gr.HTML(initial[0])
                business = gr.HTML(initial[2])
            with gr.Column(scale=4):
                matrix = gr.HTML(initial[1])
                gr.LinePlot(
                    tradeoff_data(), x="threshold", y="value", color="metric", color_map=METRIC_COLORS,
                    x_title="Decision threshold", y_title="Score (%)", y_lim=[0, 100], height=250,
                    tooltip=["threshold", "metric", "value"], show_label=False,
                )
        slider.change(threshold_view, slider, [cards, matrix, business], trigger_mode="always_last",
                      show_progress="hidden", api_name=False)
        reset.click(lambda: round(THRESHOLD, 2), None, slider, api_name=False)

    if ART.comparison is not None:
        gr.Markdown("### How the models compare")
        gr.Dataframe(comparison_display(), interactive=False, wrap=False, show_label=False)
        gr.HTML('<div class="caption">The four boosting models are within 0.0005 ROC-AUC of each other. Tuning improved '
                'the original notebook\'s XGBoost by 0.0002, less than the variation between CV folds.</div>')
    if ART.curves is not None:
        gr.Markdown("### ROC and precision-recall curves")
        curve_choice = gr.Radio([("ROC curve", "roc"), ("Precision-recall curve", "pr")], value="roc",
                                show_label=False)
        curve_plot = gr.LinePlot(curve_data("roc"), x="x", y="y", color="model", color_map=ui.MODEL_COLORS,
                                 x_title=CURVE_AXES["roc"][0], y_title=CURVE_AXES["roc"][1], y_lim=[0, 1],
                                 height=420, show_label=False, tooltip=["model", "x", "y"])
        gr.HTML('<div class="caption">ROC: higher and further left is better. Precision-recall: the gray line is the '
                'churn rate, i.e. random guessing. The four boosting models overlap almost exactly.</div>')
        curve_choice.change(
            lambda curve: gr.update(value=curve_data(curve), x_title=CURVE_AXES[curve][0], y_title=CURVE_AXES[curve][1]),
            curve_choice, curve_plot, api_name=False, show_progress="hidden")
    with gr.Row(equal_height=False):
        if ART.importance is not None:
            with gr.Column():
                gr.Markdown("### What drives predictions overall")
                gr.HTML(ui.importance_html(ART.importance))
                gr.HTML('<div class="caption">Permutation importance: how much ROC-AUC drops when a column is '
                        'shuffled on 20,000 hold-out customers.</div>')
        with gr.Column():
            gr.Markdown("### Deployed model")
            gr.Markdown(model_details_md())


# --- App ---------------------------------------------------------------------------------------------
with gr.Blocks(title="Customer Churn Predictor", theme=THEME, css=CSS) as demo:
    gr.HTML(ui.header_html(ART.metadata))
    with gr.Tabs():
        with gr.Tab("Predict"):
            predict_inputs, predict_outputs = build_predict_tab()
        with gr.Tab("Batch scoring"):
            build_batch_tab()
        with gr.Tab("Model performance"):
            build_performance_tab()
    demo.load(analyse, predict_inputs, predict_outputs, api_name=False, show_progress="hidden")

if __name__ == "__main__":
    demo.launch()
