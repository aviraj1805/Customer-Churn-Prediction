"""Train, tune and compare all models, then save the best pipeline and the evaluation reports.

    python -m src.train                          # full run (~45-60 min on 12 cores)
    python -m src.train --quick                  # smoke run on a small sample (~3 min)
    python -m src.train --models xgboost lightgbm

For each model: GridSearchCV (stratified k-fold, ROC-AUC) on a stratified sample of the training
split -> out-of-fold predictions with the best settings (CV score + F1-optimal threshold) -> refit
on the full training split -> evaluate on the untouched hold-out test set. The model with the
best cross-validated ROC-AUC is saved; the test set is never used to make any choice.
"""
import argparse
import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src import evaluate
from src.config import load_config, resolve_path
from src.data import load_dataset, stratified_sample, train_test
from src.features import RAW_FEATURES
from src.models import DISPLAY_NAMES, baseline_pipeline, candidate_pipeline

QUICK_OVERRIDES = {"train_rows": 20_000, "tuning_rows": 5_000, "cv_folds": 3}
QUICK_OUTPUT_DIR = ".quick_run"  # smoke runs never overwrite the real model or reports


def use_quick_output_dir(cfg: dict) -> None:
    """Redirect every artifact path into QUICK_OUTPUT_DIR."""
    for key in ("model", "metadata", "reports_dir", "figures_dir", "submission"):
        cfg["paths"][key] = f"{QUICK_OUTPUT_DIR}/{cfg['paths'][key]}"


def tune(model_name, X_tune, y_tune, cv, cfg):
    """Grid-search one model on the tuning sample; return the best settings and the full CV table."""
    pipeline, grid = candidate_pipeline(model_name, cfg)
    search = GridSearchCV(
        pipeline,
        grid,
        scoring=cfg["tuning"]["scoring"],
        cv=cv,
        n_jobs=cfg["tuning"]["n_jobs"],
        refit=False,  # we refit ourselves on the full training split, not the tuning sample
    )
    search.fit(X_tune, y_tune)
    best_params = {k.removeprefix("model__"): v for k, v in search.best_params_.items()}
    return best_params, search.cv_results_


def out_of_fold(pipeline, X, y, cv):
    """Cross-validated predictions: every row is scored by a model that never saw it."""
    oof = np.zeros(len(y))
    fold_scores = []
    for train_idx, val_idx in cv.split(X, y):
        model = clone(pipeline).fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
        fold_scores.append(roc_auc_score(y.iloc[val_idx], oof[val_idx]))
    return oof, np.array(fold_scores)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", help="subset of models from config.yaml (default: all)")
    parser.add_argument("--quick", action="store_true", help="small sample, 3 folds; for smoke testing")
    args = parser.parse_args(argv)

    cfg = load_config()
    if args.quick:
        use_quick_output_dir(cfg)
    model_names = args.models or list(cfg["models"])
    cv_folds = QUICK_OVERRIDES["cv_folds"] if args.quick else cfg["tuning"]["cv_folds"]
    tuning_rows = QUICK_OVERRIDES["tuning_rows"] if args.quick else cfg["tuning"]["sample_size"]
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=cfg["seed"])
    reports_dir = resolve_path(cfg["paths"]["reports_dir"])
    figures_dir = resolve_path(cfg["paths"]["figures_dir"])

    X, y = load_dataset(cfg)
    if args.quick:
        X, y = stratified_sample(X, y, QUICK_OVERRIDES["train_rows"], cfg["seed"])
    X_train, X_test, y_train, y_test = train_test(X, y, cfg)
    X_tune, y_tune = stratified_sample(X_train, y_train, tuning_rows, cfg["seed"])
    print(f"Train {len(X_train):,} rows | hold-out test {len(X_test):,} rows | "
          f"tuning sample {len(X_tune):,} rows | churn rate {y.mean():.1%}", flush=True)

    results, test_probas = {}, {}
    best_name, best_pipeline = None, None
    for name in ["baseline"] + model_names:
        started = time.perf_counter()
        if name == "baseline":
            best_params, pipeline = cfg["baseline"]["params"], baseline_pipeline(cfg)
        else:
            best_params, cv_results = tune(name, X_tune, y_tune, cv, cfg)
            evaluate.write_grid_results(name, cv_results, reports_dir)
            pipeline, _ = candidate_pipeline(name, cfg)
            pipeline.set_params(**{f"model__{k}": v for k, v in best_params.items()})

        oof, fold_scores = out_of_fold(pipeline, X_tune, y_tune, cv)
        threshold = evaluate.choose_threshold(y_tune, oof)
        pipeline.fit(X_train, y_train)
        test_probas[name] = pipeline.predict_proba(X_test)[:, 1]

        results[name] = {
            "best_params": best_params,
            "cv_auc_mean": float(fold_scores.mean()),
            "cv_auc_std": float(fold_scores.std()),
            "threshold": threshold,
            "test_metrics": evaluate.classification_metrics(y_test, test_probas[name], threshold),
            "seconds": time.perf_counter() - started,
        }
        # Keep only the best tuned pipeline in memory (a Random Forest can take hundreds of MB).
        if name != "baseline" and (best_name is None or results[name]["cv_auc_mean"] > results[best_name]["cv_auc_mean"]):
            best_name, best_pipeline = name, pipeline
        print(f"[{DISPLAY_NAMES[name]}] CV AUC {results[name]['cv_auc_mean']:.5f} | "
              f"test AUC {results[name]['test_metrics']['roc_auc']:.5f} | best {best_params} | "
              f"{results[name]['seconds']:.0f}s", flush=True)

    # Reports and figures
    table = evaluate.comparison_table(results)
    evaluate.write_comparison(table, results, reports_dir)
    evaluate.plot_model_comparison(table, figures_dir / "model_comparison.png")
    evaluate.plot_roc_curves(y_test, test_probas, figures_dir / "roc_curves.png")
    evaluate.plot_pr_curves(y_test, test_probas, figures_dir / "precision_recall_curves.png")
    test_preds = {name: (test_probas[name] >= results[name]["threshold"]).astype(int) for name in results}
    evaluate.plot_confusion_matrices(y_test, test_preds, figures_dir / "confusion_matrices.png")
    evaluate.plot_confusion_matrix(
        y_test, test_preds[best_name],
        f"{DISPLAY_NAMES[best_name]} at threshold {results[best_name]['threshold']:.3f}",
        figures_dir / "confusion_matrix_best.png",
    )
    importances = evaluate.permutation_importances(best_pipeline, X_test, y_test, cfg["seed"])
    importances.to_csv(reports_dir / "feature_importance.csv", index=False)
    evaluate.plot_feature_importance(importances, DISPLAY_NAMES[best_name], figures_dir / "feature_importance.png")

    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    save_model(best_name, best_pipeline, results[best_name], cfg, n_train=len(X_train), n_test=len(X_test))


def save_model(name: str, pipeline, result: dict, cfg: dict, n_train: int, n_test: int) -> None:
    """Save the fitted pipeline plus the metadata the app and README need."""
    model_path = resolve_path(cfg["paths"]["model"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path, compress=3)

    metadata = {
        "model_name": name,
        "display_name": DISPLAY_NAMES[name],
        "best_params": result["best_params"],
        "threshold": result["threshold"],
        "cv_roc_auc_mean": result["cv_auc_mean"],
        "cv_roc_auc_std": result["cv_auc_std"],
        "test_metrics": result["test_metrics"],
        "n_train_rows": n_train,
        "n_test_rows": n_test,
        "features": RAW_FEATURES,
        "engineered_features": cfg["features"]["engineered"],
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "library_versions": {"scikit-learn": sklearn.__version__, **_model_library_version(name)},
    }
    with open(resolve_path(cfg["paths"]["metadata"]), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved {DISPLAY_NAMES[name]} -> {model_path} ({model_path.stat().st_size / 1e6:.1f} MB), "
          f"threshold {result['threshold']:.3f}")


def _model_library_version(name: str) -> dict:
    """Version of the library needed to unpickle the model, e.g. for the deployment requirements."""
    if name == "xgboost":
        import xgboost
        return {"xgboost": xgboost.__version__}
    if name == "lightgbm":
        import lightgbm
        return {"lightgbm": lightgbm.__version__}
    return {}


if __name__ == "__main__":
    main()
