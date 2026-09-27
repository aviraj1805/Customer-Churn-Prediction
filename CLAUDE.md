# CLAUDE.md

## Purpose
Customer churn prediction for Kaggle Playground Series S6E3 (telecom customers, binary target `Churn`, metric ROC-AUC).
Refactored from a single Colab notebook into a production-style ML project: scikit-learn pipelines, tuned models,
evaluation reports, tests, and a Gradio demo on Hugging Face Spaces. Portfolio project; the owner must be able to
explain every design decision in interviews, so prefer clear, standard scikit-learn idioms over clever code.

## Structure (target)
```
data/raw/            Kaggle CSVs (gitignored; see data/README.md)
notebooks/           01 original Colab notebook (untouched), 02 cleaned EDA
src/                 config, data, features, preprocessing, models, train, evaluate, predict
models/              best_model.joblib + metadata.json (only these are tracked)
reports/             model comparison table; figures/ for all plots
submissions/         Kaggle submission files
app/                 Gradio app for Hugging Face Spaces
scripts/             deployment helper
tests/               pytest suite (uses synthetic data, no Kaggle download needed)
```

## Commands (Windows, run from repo root)
- Setup: `python -m venv .venv` then `.venv\Scripts\python -m pip install -r requirements.txt`
- Train: `.venv\Scripts\python -m src.train` (~15-20 min; `--quick` = 3-min smoke run into .quick_run/)
- Predict: `.venv\Scripts\python -m src.predict` (Kaggle submission) or `--input x.csv --output y.csv`
- Test: `.venv\Scripts\python -m pytest`
- App: `.venv\Scripts\python app/app.py`
- Deploy: `.venv\Scripts\python scripts/deploy_space.py`

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
- [ ] Task 8: cleaned EDA notebook
- [ ] Task 9: Gradio app
- [ ] Task 10: deploy to Hugging Face Spaces
- [ ] Task 11: README + docs
- [ ] Task 12: final verification, merge to main, push
