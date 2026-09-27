# Experiments log

## 1. Pipeline parity with the original notebook

The original notebook's XGBoost settings (`n_estimators=300, learning_rate=0.05, max_depth=6`) were
re-run through the new scikit-learn pipeline on the **same 5 stratified folds** (seed 42).

| Fold | Original notebook | New pipeline |
|---|---|---|
| 1 | 0.91539 | 0.91555 |
| 2 | 0.91648 | 0.91651 |
| 3 | 0.91581 | 0.91583 |
| 4 | 0.91689 | 0.91694 |
| 5 | 0.91415 | 0.91427 |
| **Mean** | **0.91574** | **0.91582** |

The refactor reproduces the original result (every fold within 0.0002). The small difference most likely
comes from keeping the "No internet service" / "No phone service" levels as their own one-hot columns instead of
folding them into "No" as the notebook did.

## 2. Engineered features ablation

Features: `AvgChargePerMonth` (TotalCharges / tenure), `ChargeIncrease` (MonthlyCharges - AvgChargePerMonth),
`NumAddonServices` (count of internet add-ons). Same 5 folds as above.

| Model | Without | With | Folds improved |
|---|---|---|---|
| XGBoost (notebook params) | 0.91582 | 0.91593 | 5 / 5 |
| Logistic Regression | 0.90795 | 0.90795 | - |

Decision: keep them (`features.engineered: true`). The gain is small but consistent across every fold
and costs nothing at inference time, because the features are computed inside the pipeline.

Reproduce: `python -m scripts.feature_ablation`
