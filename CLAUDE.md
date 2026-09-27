# CLAUDE.md

## Purpose
Customer churn prediction for Kaggle Playground Series S6E3 (telecom customers, binary target `Churn`, metric ROC-AUC).
Refactored from a single Colab notebook into a production-style ML project: scikit-learn pipelines, tuned models,
evaluation reports, tests, and a Gradio demo on Render (free tier). Portfolio project; the owner must be able to
explain every design decision in interviews, so prefer clear, standard scikit-learn idioms over clever code.

## Structure (target)
```
data/raw/            Kaggle CSVs (gitignored; see data/README.md)
notebooks/           01 original Colab notebook (untouched), 02 cleaned EDA
src/                 config, data, features, preprocessing, models, train, evaluate, predict
models/              best_model.joblib + metadata.json (only these are tracked)
reports/             model comparison table; figures/ for all plots
submissions/         Kaggle submission files
app/                 Gradio app + its pinned inference requirements (deployed via render.yaml)
scripts/             feature_ablation.py (parity check vs original notebook)
tests/               pytest suite (uses synthetic data, no Kaggle download needed)
```

## Commands (Windows, run from repo root)
- Setup: `python -m venv .venv` then `.venv\Scripts\python -m pip install -r requirements.txt`
- Train: `.venv\Scripts\python -m src.train` (~15-20 min; `--quick` = 3-min smoke run into .quick_run/)
- Predict: `.venv\Scripts\python -m src.predict` (Kaggle submission) or `--input x.csv --output y.csv`
- Test: `.venv\Scripts\python -m pytest`
- App: `.venv\Scripts\python app/app.py`
- Deploy: push to `main`; Render auto-deploys from `render.yaml` (free plan, kept awake by a 5-min uptime ping).
  Keep `app/requirements.txt` pinned to the training versions (tests/test_deployment.py enforces this).
- Reproduce notebook baseline: `.venv\Scripts\python -m scripts.feature_ablation`

## Conventions
- Constraints: no new frameworks beyond the existing stack (pandas, scikit-learn, xgboost, lightgbm, matplotlib/seaborn);
  gradio only for deployment. No Streamlit.
- All paths, seeds, and hyperparameter grids live in `config.yaml`; code reads them via `src/config.py`.
- Preprocessing lives inside the saved sklearn `Pipeline`; never transform data outside it for training or inference.
- Random seed 42 everywhere. Splits and CV are stratified.
- Commits: conventional commit messages, one per completed task, author = repo-local git identity.
  No co-author trailers and no mention of Claude in commit messages.
- Workflow: one task at a time -> implement -> run/test -> verify -> commit.

## Status
- [x] Task 0: setup (branch `refactor/ml-pipeline`, .gitignore, .venv, attribution settings)
- [x] Task 1: restructure folders, data, requirements, config
- [x] Task 2: data module + tests
- [x] Task 3: features + preprocessing pipeline + tests (parity 0.91582 vs notebook 0.91574; engineered features on)
- [x] Task 4: model registry + grids
- [x] Task 5: training entry point (`--quick` writes to .quick_run/, ~3 min)
- [x] Task 6: evaluation + full training run (best: tuned XGBoost, test AUC 0.9165; reruns reproduce exactly)
- [x] Task 7: predict module + tests
- [x] Task 8: cleaned EDA notebook (executed with outputs; figures in reports/figures/eda_*.png)
- [x] Task 9: Gradio app (app/app.py; screenshot in reports/figures/app_screenshot.png)
- [~] Task 10: deploy (HF free Gradio Spaces now need PRO -> switched to Render free; render.yaml ready, live URL pending)
- [x] Task 11: README + docs
- [x] Task 12: final verification, merge to main, push
- UI overhaul (branch `feature/ui-overhaul`):
  - [x] Task 13: report tables for the app (threshold_analysis, curves, segment_churn_rates) + retrain (identical metrics)
  - [x] Task 14: src/explain.py (SHAP drivers, retention what-ifs, tenure outlook)
  - [ ] Task 15+16: app restructure (theme, components, artifacts), live Predict tab, Batch scoring tab
  - [ ] Task 17: Model performance tab (threshold explorer)
  - [ ] Task 18: Data insights + About/API tabs, JSON API
  - [ ] Task 19: polish, memory check, screenshots, docs, merge + push
