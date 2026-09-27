"""HTML building blocks for the app. Pure functions (data in, HTML string out), so they are easy to test."""
import json
import math
from html import escape

import pandas as pd

from src.predict import risk_tier

REPO_URL = "https://github.com/aviraj1805/Customer-Churn-Prediction"
APP_URL = "https://churn-predictor-aviraj.onrender.com"

FIELD_LABELS = {
    "gender": "Gender", "SeniorCitizen": "Senior citizen", "Partner": "Has a partner",
    "Dependents": "Has dependents", "tenure": "Tenure (months)", "Contract": "Contract",
    "PaperlessBilling": "Paperless billing", "PaymentMethod": "Payment method",
    "MonthlyCharges": "Monthly charges ($)", "TotalCharges": "Total charges ($)",
    "PhoneService": "Phone service", "MultipleLines": "Multiple lines", "InternetService": "Internet service",
    "OnlineSecurity": "Online security", "OnlineBackup": "Online backup", "DeviceProtection": "Device protection",
    "TechSupport": "Tech support", "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming movies",
    "AvgChargePerMonth": "Avg. charge per month (derived)", "ChargeIncrease": "Recent price change (derived)",
    "NumAddonServices": "Number of add-ons (derived)", "TenureBand": "Tenure band (months)",
}
MODEL_COLORS = {  # same palette as src/plot_style.py; the untuned baseline and chance recede in gray
    "Logistic Regression": "#2a78d6", "Random Forest": "#eb6834", "HistGradientBoosting": "#1baf7a",
    "XGBoost (tuned)": "#eda100", "LightGBM (tuned)": "#e87ba4", "XGBoost (original notebook)": "#898781",
    "Random guess": "#c3c2b7",
}
TIER_COLORS = {"Low": "var(--risk-low)", "Moderate": "var(--risk-moderate)",
               "High": "var(--risk-high)", "Very high": "var(--risk-very-high)"}


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def format_value(field: str, value) -> str:
    """Human-readable input value, e.g. 85.5 -> '$85.50' for charges, 1 -> 'Yes' for senior citizen."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if field in ("MonthlyCharges", "TotalCharges", "AvgChargePerMonth"):
        return f"${value:,.2f}"
    if field == "ChargeIncrease":
        return f"{'+' if value >= 0 else '-'}${abs(value):,.2f}"
    if field == "SeniorCitizen":
        return "Yes" if value else "No"
    if field in ("tenure", "NumAddonServices"):
        return f"{value:.0f}"
    return str(value)


def header_html(metadata: dict) -> str:
    m = metadata["test_metrics"]
    chips = [("ROC-AUC", f"{m['roc_auc']:.3f}"), ("Recall", pct(m["recall"])),
             ("Precision", pct(m["precision"])), ("Customers trained on", f"{metadata['n_train_rows']:,}")]
    chip_html = "".join(f'<div class="chip"><div class="v">{v}</div><div class="l">{escape(l)}</div></div>'
                        for l, v in chips)
    return f"""
<div class="hero">
  <div>
    <h1>Customer Churn Predictor</h1>
    <p>Score a telecom customer's risk of leaving, see what drives it, and test retention actions. Results update live.</p>
    <div class="links">Model: {escape(metadata['display_name'])} · hold-out test on {metadata['n_test_rows']:,} customers ·
      <a href="{REPO_URL}" target="_blank">Source code on GitHub</a></div>
  </div>
  <div class="chips">{chip_html}</div>
</div>"""


def badge_html(tier: str) -> str:
    return f'<span class="badge" style="--tier:{TIER_COLORS[tier]}"><span class="dot"></span>{tier} risk</span>'


def gauge_html(proba: float, threshold: float) -> str:
    """Semicircle gauge: arc = probability, tick = decision threshold."""
    radius, cx, cy = 80, 100, 96
    length = math.pi * radius
    tier = risk_tier(proba, threshold)
    angle = math.pi * (1 - threshold)
    x1, y1 = cx + 70 * math.cos(angle), cy - 70 * math.sin(angle)
    x2, y2 = cx + 90 * math.cos(angle), cy - 90 * math.sin(angle)
    arc = f"M {cx - radius} {cy} A {radius} {radius} 0 0 1 {cx + radius} {cy}"
    return f"""
<div class="gauge" role="img" aria-label="Churn probability {pct(proba)}, {tier} risk">
  <svg viewBox="0 0 200 118">
    <path d="{arc}" fill="none" stroke="var(--border-color-primary)" stroke-width="12" stroke-linecap="round"/>
    <path class="arc" d="{arc}" fill="none" stroke="{TIER_COLORS[tier]}" stroke-width="12" stroke-linecap="round"
      stroke-dasharray="{length:.2f}" stroke-dashoffset="{length * (1 - proba):.2f}" style="--arc-len:{length:.2f}"/>
    <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="var(--body-text-color)" stroke-width="2.5"/>
    <text x="{x2:.1f}" y="{y2 - 4:.1f}" text-anchor="middle" class="sub">threshold {pct(threshold, 0)}</text>
    <text x="{cx}" y="{cy - 14}" text-anchor="middle" class="pct">{pct(proba)}</text>
    <text x="{cx}" y="{cy + 2}" text-anchor="middle" class="sub">churn probability</text>
    <text x="{cx - radius}" y="{cy + 18}" text-anchor="middle" class="sub">0%</text>
    <text x="{cx + radius}" y="{cy + 18}" text-anchor="middle" class="sub">100%</text>
  </svg>
</div>"""


def verdict_html(proba: float, threshold: float, churn_rate: float) -> str:
    tier = risk_tier(proba, threshold)
    if proba >= threshold:
        headline = "Likely to churn: flag for a retention offer"
    else:
        headline = "Unlikely to churn: no action needed"
    ratio = proba / churn_rate
    comparison = f"{ratio:.1f}x the average churn rate ({pct(churn_rate)})"
    return f"""
<div class="verdict">
  {badge_html(tier)}
  <div class="headline">{headline}</div>
  <div class="detail">{comparison}. Customers at or above the {pct(threshold)} threshold are flagged.</div>
</div>"""


def drivers_html(table: pd.DataFrame | None, top: int = 8) -> str:
    """Diverging bars: red pushes this customer's risk up, blue pushes it down."""
    if table is None:
        return '<div class="caption">Per-customer explanations are available for tree-boosting models only.</div>'
    rows = table.head(top)
    scale = rows["contribution"].abs().max() or 1
    html = ['<div class="legend"><span><i style="background:var(--pushes-up)"></i>Raises risk</span>'
            '<span><i style="background:var(--pushes-down)"></i>Lowers risk</span></div><div class="bars">']
    for row in rows.itertuples():
        label = FIELD_LABELS.get(row.field, row.field)
        value = format_value(row.field, row.value)
        direction = "up" if row.contribution >= 0 else "down"
        width = 50 * abs(row.contribution) / scale
        tooltip = f"{label} = {value}: {row.contribution:+.2f} log-odds ({'raises' if direction == 'up' else 'lowers'} risk)"
        html.append(
            f'<div class="bar-row" title="{escape(tooltip)}"><div class="name">{escape(label)}<small>{escape(value)}</small></div>'
            f'<div class="track diverging"><div class="bar {direction}" style="width:{width:.1f}%"></div></div>'
            f'<div class="num">{row.contribution:+.2f}</div></div>'
        )
    html.append("</div>")
    return "".join(html)


def scenarios_html(table: pd.DataFrame, base_proba: float) -> str:
    if table.empty:
        return ('<div class="caption">This customer already has the lowest-risk contract, payment method and '
                'add-ons the model knows about.</div>')
    rows = []
    for row in table.itertuples():
        direction = "down" if row.change < 0 else "up"
        rows.append(
            f'<div class="scenario"><div>{escape(row.action)}</div>'
            f'<div class="p">{pct(base_proba)} &rarr; <b>{pct(row.churn_probability)}</b></div>'
            f'<div class="delta {direction}">{row.change * 100:+.1f} pts</div></div>'
        )
    return "".join(rows)


def kpi_cards_html(cards: list[tuple[str, str, str]]) -> str:
    """Cards of (label, value, subtitle)."""
    return '<div class="kpis">' + "".join(
        f'<div class="kpi"><div class="l">{escape(l)}</div><div class="v">{v}</div><div class="s">{escape(s)}</div></div>'
        for l, v, s in cards) + "</div>"


def confusion_html(row: pd.Series) -> str:
    """2x2 confusion matrix from a threshold-table row (tp, fp, fn, tn)."""
    total = row["tp"] + row["fp"] + row["fn"] + row["tn"]

    def cell(n, title, kind):
        return f'<div class="cell {kind}"><div class="n">{int(n):,}</div><div class="t">{title} · {n / total:.1%}</div></div>'

    return (
        '<div class="cm"><div></div><div class="h">Predicted: stays</div><div class="h">Predicted: churns</div>'
        f'<div class="h">Actually stays</div>{cell(row["tn"], "correctly cleared", "good")}{cell(row["fp"], "false alarm", "bad")}'
        f'<div class="h">Actually churns</div>{cell(row["fn"], "missed churner", "bad")}{cell(row["tp"], "caught churner", "good")}'
        "</div>"
    )


def business_html(row: pd.Series, per: int = 10_000) -> str:
    """The threshold's consequences for a campaign, scaled to ``per`` customers."""
    total = row["tp"] + row["fp"] + row["fn"] + row["tn"]
    scale = per / total
    churners = (row["tp"] + row["fn"]) * scale
    return f"""
<div class="business">
  Out of <b>{per:,}</b> customers (about <b>{churners:,.0f}</b> of whom will churn), this threshold would:<br>
  &bull; flag <b>{(row["tp"] + row["fp"]) * scale:,.0f}</b> customers for a retention offer<br>
  &bull; catch <b>{row["tp"] * scale:,.0f}</b> churners ({row["recall"]:.0%} of them)<br>
  &bull; send <b>{row["fp"] * scale:,.0f}</b> offers to customers who would have stayed<br>
  &bull; miss <b>{row["fn"] * scale:,.0f}</b> churners
</div>"""


def importance_html(table: pd.DataFrame) -> str:
    """Horizontal bars for permutation importance (drop in ROC-AUC when a column is shuffled)."""
    scale = table["importance"].max() or 1
    rows = []
    for row in table.itertuples():
        label = FIELD_LABELS.get(row.feature, row.feature)
        width = 100 * max(row.importance, 0) / scale
        rows.append(
            f'<div class="bar-row" title="{escape(label)}: ROC-AUC drops by {row.importance:.4f} when shuffled">'
            f'<div class="name">{escape(label)}</div><div class="track"><div class="bar plain" style="width:{width:.1f}%"></div></div>'
            f'<div class="num">{row.importance:.4f}</div></div>'
        )
    return '<div class="bars">' + "".join(rows) + "</div>"


def about_md(metadata: dict, importance: pd.DataFrame | None, example: dict) -> str:
    """How it works, a model card and API usage, filled in from the deployed model's metadata."""
    m = metadata["test_metrics"]
    gender = ""
    if importance is not None:
        imp = importance.set_index("feature")["importance"]
        gender = (f"- **Fairness check:** gender has no measurable influence on predictions (permutation importance "
                  f"{imp.get('gender', 0):.4f}); senior-citizen status has a small one "
                  f"({imp.get('SeniorCitizen', 0):.4f}).\n")
    payload = json.dumps(example, indent=2)
    return f"""
### How it works
1. **Data:** 594,194 customers from the [Kaggle Playground S6E3](https://www.kaggle.com/competitions/playground-series-s6e3)
   competition (synthetic telecom data, CC BY 4.0), of whom {metadata.get('churn_rate', 0.225):.1%} churned.
2. **One pipeline:** imputation, scaling, one-hot encoding and three engineered features live inside the saved
   scikit-learn pipeline, so this app applies exactly the steps used in training.
3. **Five models** (Logistic Regression, Random Forest, HistGradientBoosting, XGBoost, LightGBM) tuned with
   `GridSearchCV` and stratified 5-fold cross-validation on ROC-AUC. The best cross-validated model is deployed.
4. **Decision threshold** ({metadata['threshold']:.3f}) chosen to maximise F1 on out-of-fold training predictions.
5. **Honest evaluation** on {metadata['n_test_rows']:,} held-out customers that were never used for any choice.
6. **Explanations** come from the model's built-in SHAP values. What-if scenarios re-score the customer with one change.

### Model card
- **Model:** {metadata['display_name']}, trained {metadata['trained_at'][:10]} on {metadata['n_train_rows']:,} customers.
- **Intended use:** ranking customers for retention outreach, as decision support for a retention team.
- **Performance (hold-out):** ROC-AUC {m['roc_auc']:.4f}, recall {m['recall']:.1%}, precision {m['precision']:.1%}, F1 {m['f1']:.3f}.
- **Limitations:** the data is synthetic, so probabilities will not transfer to a real company without retraining on its
  own data. What-if scenarios show what the model associates with lower risk, not proven causal effects, and they
  keep the monthly bill fixed.
{gender}- **Maintenance:** retrain when the churn rate or the customer mix drifts; the decision threshold should be
  re-chosen with the retention team's budget in mind.

### Use the API
Every prediction in this app is also available as a JSON API (endpoint `/predict`).

```python
from gradio_client import Client

client = Client("{APP_URL}/")
result = client.predict(customer={payload}, api_name="/predict")
print(result["churn_probability"], result["risk_tier"], result["top_drivers"][:2])
```

With cURL (two steps: submit, then read the result):

```bash
EVENT_ID=$(curl -s -X POST {APP_URL}/gradio_api/call/predict \\
  -H "Content-Type: application/json" \\
  -d '{{"data": [{json.dumps(example)}]}}' | python -c "import sys, json; print(json.load(sys.stdin)['event_id'])")
curl -s {APP_URL}/gradio_api/call/predict/$EVENT_ID
```

### Tech stack
Python · pandas · scikit-learn · XGBoost · LightGBM · Gradio · Render · pytest ·
[source code, training pipeline and tests on GitHub]({REPO_URL})
"""
