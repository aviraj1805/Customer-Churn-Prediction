"""Train, tune and compare all models, then save the best pipeline.

    python -m src.train                          # full run (~30-60 min)
    python -m src.train --quick                  # smoke run on a small sample (~1 min)
    python -m src.train --models xgboost lightgbm

Steps: load data -> stratified 80/20 split -> GridSearchCV (stratified k-fold, ROC-AUC) per model
on a stratified sample of the training split -> refit the best settings on the full training split
-> evaluate every model on the untouched hold-out set -> save the best pipeline and its metadata.
"""
import argparse
import json
import time
from datetime import datetime, timezone

import joblib
import sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold

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


def tune(model_name, X_tune, y_tune, cfg, cv_folds):
    """Grid-search one model; return the best settings, their CV score and the full CV table."""
    pipeline, grid = candidate_pipeline(model_name, cfg)
    search = GridSearchCV(
        pipeline,
        grid,
        scoring=cfg["tuning"]["scoring"],
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=cfg["seed"]),
        n_jobs=cfg["tuning"]["n_jobs"],
        refit=False,  # we refit ourselves on the full training split, not the tuning sample
    )
    search.fit(X_tune, y_tune)
    best = search.best_index_
    return {
        "best_params": {k.removeprefix("model__"): v for k, v in search.best_params_.items()},
        "cv_auc_mean": float(search.cv_results_["mean_test_score"][best]),
        "cv_auc_std": float(search.cv_results_["std_test_score"][best]),
        "n_candidates": len(search.cv_results_["params"]),
    }


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

    X, y = load_dataset(cfg)
    if args.quick:
        X, y = stratified_sample(X, y, QUICK_OVERRIDES["train_rows"], cfg["seed"])
    X_train, X_test, y_train, y_test = train_test(X, y, cfg)
    X_tune, y_tune = stratified_sample(X_train, y_train, tuning_rows, cfg["seed"])
    print(f"Train {len(X_train):,} rows | hold-out test {len(X_test):,} rows | "
          f"tuning sample {len(X_tune):,} rows | churn rate {y.mean():.1%}")

    results = {}
    for name in ["baseline"] + model_names:
        started = time.perf_counter()
        if name == "baseline":
            tuning = {"best_params": cfg["baseline"]["params"], "cv_auc_mean": None,
                      "cv_auc_std": None, "n_candidates": 0}
            pipeline = baseline_pipeline(cfg)
        else:
            tuning = tune(name, X_tune, y_tune, cfg, cv_folds)
            pipeline, _ = candidate_pipeline(name, cfg)
            pipeline.set_params(**{f"model__{k}": v for k, v in tuning["best_params"].items()})
        pipeline.fit(X_train, y_train)
        results[name] = {**tuning, "pipeline": pipeline, "seconds": time.perf_counter() - started}

        cv = f"CV AUC {tuning['cv_auc_mean']:.5f}" if tuning["cv_auc_mean"] is not None else "untuned"
        print(f"[{DISPLAY_NAMES[name]}] {cv} | best {tuning['best_params']} | "
              f"{results[name]['seconds']:.0f}s", flush=True)

    save_best(results, cfg, model_names)


def save_best(results: dict, cfg: dict, model_names: list[str]) -> None:
    """Pick the tuned model with the highest CV AUC and save its pipeline and metadata."""
    best_name = max(model_names, key=lambda n: results[n]["cv_auc_mean"])
    best = results[best_name]

    model_path = resolve_path(cfg["paths"]["model"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["pipeline"], model_path, compress=3)

    metadata = {
        "model_name": best_name,
        "display_name": DISPLAY_NAMES[best_name],
        "best_params": best["best_params"],
        "cv_auc_mean": best["cv_auc_mean"],
        "cv_auc_std": best["cv_auc_std"],
        "features": RAW_FEATURES,
        "engineered_features": cfg["features"]["engineered"],
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
    }
    with open(resolve_path(cfg["paths"]["metadata"]), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    size_mb = model_path.stat().st_size / 1e6
    print(f"Best model: {DISPLAY_NAMES[best_name]} -> {model_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
