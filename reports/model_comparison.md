| Model | CV ROC-AUC | Test ROC-AUC | Test PR-AUC | Accuracy | Precision | Recall | F1 | Threshold |
|---|---|---|---|---|---|---|---|---|
| XGBoost (tuned) | 0.9141 ± 0.0012 | **0.9165** | 0.7552 | 0.8504 | 0.6374 | 0.7788 | 0.7011 | 0.362 |
| XGBoost (original notebook) | 0.9136 ± 0.0012 | **0.9163** | 0.7546 | 0.8498 | 0.6356 | 0.7804 | 0.7006 | 0.359 |
| LightGBM (tuned) | 0.9136 ± 0.0013 | **0.9161** | 0.7544 | 0.8455 | 0.6216 | 0.8028 | 0.7007 | 0.334 |
| HistGradientBoosting | 0.9135 ± 0.0013 | **0.9161** | 0.7536 | 0.8485 | 0.6315 | 0.7860 | 0.7003 | 0.651 |
| Random Forest | 0.9122 ± 0.0012 | **0.9144** | 0.7493 | 0.8465 | 0.6268 | 0.7876 | 0.6980 | 0.355 |
| Logistic Regression | 0.9072 ± 0.0015 | **0.9084** | 0.7270 | 0.8368 | 0.6031 | 0.8051 | 0.6896 | 0.339 |

## Best hyperparameters

- **XGBoost (original notebook)**: `n_estimators=300`, `learning_rate=0.05`, `max_depth=6`, `eval_metric=auc`, `n_jobs=-1`
- **Logistic Regression**: `C=1.0`, `class_weight=None`
- **Random Forest**: `class_weight=None`, `max_depth=16`, `min_samples_leaf=20`
- **HistGradientBoosting**: `class_weight=balanced`, `l2_regularization=1.0`, `learning_rate=0.05`, `max_leaf_nodes=31`
- **XGBoost (tuned)**: `learning_rate=0.05`, `max_depth=4`, `n_estimators=600`, `scale_pos_weight=1.0`
- **LightGBM (tuned)**: `class_weight=None`, `learning_rate=0.05`, `n_estimators=300`, `num_leaves=31`
