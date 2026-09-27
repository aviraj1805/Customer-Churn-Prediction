"""HTML building blocks for the app. Pure functions (data in, HTML string out), so they are easy to test."""
import math
from html import escape

import pandas as pd

from src.predict import risk_tier

REPO_URL = "https://github.com/aviraj1805/Customer-Churn-Prediction"

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
