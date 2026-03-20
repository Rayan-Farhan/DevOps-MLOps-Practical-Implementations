"""
Stage 4: Model Training with MLflow Experiment Tracking

Training flow per run:
  1. StratifiedKFold cross-validation on all four candidate models
  2. Select the best candidate by CV primary metric
  3. If tuning.enabled in params.yaml: GridSearchCV on the winner's param_grid
  4. Refit the (possibly tuned) winner on the full training set
  5. Log everything to an MLflow run
  6. Register the best model in the MLflow Model Registry

Candidate models:
  - RandomForestClassifier
  - LogisticRegression
  - GradientBoostingClassifier
  - XGBClassifier

Outputs:
  data/processed/trained_model.pkl    -- best model bundle {model, scaler, features, …}
  data/processed/training_report.json

MLflow UI:
  mlflow ui --backend-store-uri sqlite:///mlflow.db
  → http://localhost:5000
  Experiment: "diabetes-prediction"
"""

import json
import logging
import os
import pickle
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import yaml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
EXPERIMENT_NAME = "diabetes-prediction"
REGISTRY_MODEL_NAME = "diabetes-classifier"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_params() -> dict:
    with open(ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


def load_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.load(PROCESSED_DIR / "X_train.npy")
    y = np.load(PROCESSED_DIR / "y_train.npy")
    logger.info("Training data loaded: X=%s  y=%s", X.shape, y.shape)
    return X, y


def load_feature_names() -> list[str]:
    with open(PROCESSED_DIR / "feature_names.json") as f:
        return json.load(f)


def load_scaler():
    with open(PROCESSED_DIR / "scaler.pkl", "rb") as f:
        return pickle.load(f)


def load_train_medians() -> dict:
    with open(PROCESSED_DIR / "train_medians.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "f1_diabetic": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_macro":    float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc":     float(roc_auc_score(y_true, y_pred)),
    }


# ---------------------------------------------------------------------------
# Model catalogue (built from params.yaml)
# ---------------------------------------------------------------------------

def build_models(params: dict) -> dict:
    rf  = params["models"]["random_forest"]
    lr  = params["models"]["logistic_regression"]
    gb  = params["models"]["gradient_boosting"]
    xgb = params["models"]["xgboost"]

    return {
        "random_forest": RandomForestClassifier(
            n_estimators=rf["n_estimators"],
            max_depth=rf["max_depth"],
            min_samples_split=rf.get("min_samples_split", 2),
            min_samples_leaf=rf.get("min_samples_leaf", 1),
            random_state=rf["random_state"],
        ),
        "logistic_regression": LogisticRegression(
            C=lr["C"],
            max_iter=lr["max_iter"],
            random_state=lr["random_state"],
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=gb["n_estimators"],
            learning_rate=gb["learning_rate"],
            max_depth=gb["max_depth"],
            random_state=gb["random_state"],
        ),
        "xgboost": XGBClassifier(
            n_estimators=xgb["n_estimators"],
            learning_rate=xgb["learning_rate"],
            max_depth=xgb["max_depth"],
            random_state=xgb["random_state"],
            eval_metric="logloss",
            verbosity=0,
        ),
    }


# ---------------------------------------------------------------------------
# Step 1: Cross-validation screening
# ---------------------------------------------------------------------------

SCORER_MAP = {
    "f1_diabetic": "f1",
    "f1_macro":    "f1_macro",
    "accuracy":    "accuracy",
    "roc_auc":     "roc_auc",
}


def run_cross_validation(
    models: dict,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int,
    primary_metric: str,
) -> list[dict]:
    """
    StratifiedKFold CV for every candidate model.
    Returns list sorted by mean CV primary metric (descending).
    """
    skf    = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scorer = SCORER_MAP.get(primary_metric, "f1")

    cv_results = []
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=skf, scoring=scorer, n_jobs=-1)
        mean_score, std_score = float(scores.mean()), float(scores.std())
        logger.info(
            "CV [%-20s]  %s = %.4f ± %.4f",
            name, primary_metric, mean_score, std_score,
        )
        cv_results.append({
            "name":    name,
            "model":   model,
            "cv_mean": mean_score,
            "cv_std":  std_score,
        })

    cv_results.sort(key=lambda r: r["cv_mean"], reverse=True)
    return cv_results


# ---------------------------------------------------------------------------
# Step 2: Optional hyperparameter tuning on the best candidate
# ---------------------------------------------------------------------------

def tune_model(
    name: str,
    model,
    param_grids: dict,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int,
    primary_metric: str,
) -> tuple:
    """
    GridSearchCV on the best CV candidate using its param_grid from params.yaml.
    Returns (best_estimator, best_params).
    If no param_grid is defined for `name`, returns the original model unchanged.
    """
    grid = param_grids.get(name)
    if not grid:
        logger.info("No param_grid for '%s' — skipping GridSearchCV.", name)
        return model, {}

    scorer = SCORER_MAP.get(primary_metric, "f1")
    skf    = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    logger.info("GridSearchCV on '%s' with grid: %s", name, grid)

    gs = GridSearchCV(model, grid, scoring=scorer, cv=skf, n_jobs=-1, refit=True)
    gs.fit(X, y)
    logger.info(
        "GridSearchCV best params: %s   best CV score=%.4f",
        gs.best_params_, gs.best_score_,
    )
    return gs.best_estimator_, gs.best_params_


# ---------------------------------------------------------------------------
# Step 3: Final fit + MLflow logging
# ---------------------------------------------------------------------------

def fit_and_log(
    name: str,
    model,
    model_params: dict,
    cv_mean: float,
    cv_std: float,
    tuned_params: dict,
    X: np.ndarray,
    y: np.ndarray,
    primary_metric: str,
    already_fitted: bool = False,
) -> tuple[object, dict, str]:
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tag("model_type", name)
        mlflow.log_params(model_params)
        if tuned_params:
            mlflow.log_params({f"tuned_{k}": v for k, v in tuned_params.items()})
        mlflow.log_metrics({
            f"cv_mean_{primary_metric}": cv_mean,
            f"cv_std_{primary_metric}":  cv_std,
        })

        if not already_fitted:
            model.fit(X, y)

        y_pred  = model.predict(X)
        metrics = compute_metrics(y, y_pred)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")

        logger.info(
            "[%-20s] accuracy=%.4f  f1_diabetic=%.4f  roc_auc=%.4f  "
            "cv_%s=%.4f±%.4f",
            name, metrics["accuracy"], metrics["f1_diabetic"],
            metrics["roc_auc"], primary_metric, cv_mean, cv_std,
        )
        logger.info(
            "\n%s",
            classification_report(y, y_pred, target_names=["Non-Diabetic", "Diabetic"]),
        )
        run_id = run.info.run_id

    return model, metrics, run_id


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    params = load_params()
    primary_metric = params["evaluation"]["primary_metric"]
    tuning_cfg     = params.get("tuning", {})
    tuning_enabled = tuning_cfg.get("enabled", False)
    cv_folds       = tuning_cfg.get("cv_folds", 5)
    param_grids    = tuning_cfg.get("param_grids", {})

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}",
    )
    mlflow.set_tracking_uri(tracking_uri)
    logger.info("MLflow tracking URI: %s", tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, y_train = load_data()
    feature_names    = load_feature_names()
    scaler           = load_scaler()
    train_medians    = load_train_medians()
    zero_impute_cols = params.get("data", {}).get("replace_zeros_with_nan", [])
    models           = build_models(params)

    model_params_map = {
        "random_forest":       params["models"]["random_forest"],
        "logistic_regression": params["models"]["logistic_regression"],
        "gradient_boosting":   params["models"]["gradient_boosting"],
        "xgboost":             params["models"]["xgboost"],
    }

    # ── Step 1: Cross-validation ─────────────────────────────────────────
    logger.info("=== Step 1: StratifiedKFold CV (%d folds) ===", cv_folds)
    cv_results  = run_cross_validation(models, X_train, y_train, cv_folds, primary_metric)
    best_cv     = cv_results[0]
    winner_name = best_cv["name"]
    logger.info("Best CV candidate: %s (cv_%s=%.4f)", winner_name, primary_metric, best_cv["cv_mean"])

    # ── Step 2: Hyperparameter tuning ────────────────────────────────────
    winner_model = best_cv["model"]
    tuned_params = {}
    if tuning_enabled:
        logger.info("=== Step 2: GridSearchCV on '%s' ===", winner_name)
        winner_model, tuned_params = tune_model(
            winner_name, winner_model, param_grids,
            X_train, y_train, cv_folds, primary_metric,
        )
    else:
        logger.info("Tuning disabled — skipping GridSearchCV.")

    # ── Step 3: Log all candidates to MLflow ─────────────────────────────
    logger.info("=== Step 3: MLflow logging ===")
    all_results = []
    for r in cv_results:
        name          = r["name"]
        is_winner     = name == winner_name
        model         = winner_model if is_winner else r["model"]
        already_fit   = is_winner and bool(tuned_params)

        fitted, metrics, run_id = fit_and_log(
            name, model, model_params_map[name],
            r["cv_mean"], r["cv_std"],
            tuned_params if is_winner else {},
            X_train, y_train, primary_metric,
            already_fitted=already_fit,
        )
        all_results.append({
            "name":    name,
            "run_id":  run_id,
            "metrics": metrics,
            "cv_mean": r["cv_mean"],
            "model":   fitted,
            "is_best": is_winner,
        })

    best_result = next(r for r in all_results if r["is_best"])

    # ── Step 4: MLflow Model Registry ────────────────────────────────────
    logger.info("=== Step 4: MLflow Model Registry ===")
    registry_version = None
    try:
        model_uri = f"runs:/{best_result['run_id']}/model"
        mv = mlflow.register_model(model_uri, REGISTRY_MODEL_NAME)
        registry_version = mv.version
        logger.info(
            "Registered '%s' v%s in MLflow Model Registry",
            REGISTRY_MODEL_NAME, registry_version,
        )
    except Exception as e:
        logger.warning("Model Registry registration failed (non-fatal): %s", e)

    # ── Step 5: Save bundle + report ─────────────────────────────────────
    bundle = {
        "model":             best_result["model"],
        "scaler":            scaler,
        "features":          feature_names,
        "model_type":        winner_name,
        "train_metrics":     best_result["metrics"],
        "cv_mean":           best_result["cv_mean"],
        "tuned_params":      tuned_params,
        "mlflow_run_id":     best_result["run_id"],
        "registry_version":  registry_version,
        "train_medians":     train_medians,
        "zero_impute_cols":  zero_impute_cols,
    }
    bundle_path = PROCESSED_DIR / "trained_model.pkl"
    with open(bundle_path, "wb") as f:
        pickle.dump(bundle, f)
    logger.info("CV-winner bundle saved → %s", bundle_path)

    # Save every candidate individually so evaluate.py can pick the best
    # performer on the held-out test set (CV winner ≠ necessarily best on test).
    for r in all_results:
        individual = {
            "model":            r["model"],
            "scaler":           scaler,
            "features":         feature_names,
            "model_type":       r["name"],
            "train_metrics":    r["metrics"],
            "cv_mean":          r["cv_mean"],
            "mlflow_run_id":    r["run_id"],
            "train_medians":    train_medians,
            "zero_impute_cols": zero_impute_cols,
        }
        p = PROCESSED_DIR / f"model_{r['name']}.pkl"
        with open(p, "wb") as f:
            pickle.dump(individual, f)
        logger.info("Candidate bundle saved → %s", p)

    report = {
        "best_model":         winner_name,
        "primary_metric":     primary_metric,
        "tuning_enabled":     tuning_enabled,
        "tuned_params":       tuned_params,
        "train_metrics":      best_result["metrics"],
        "cv_mean":            best_result["cv_mean"],
        "mlflow_run_id":      best_result["run_id"],
        "mlflow_experiment":  EXPERIMENT_NAME,
        "registry_version":   registry_version,
        "all_results": [
            {
                "name":    r["name"],
                "cv_mean": r["cv_mean"],
                "metrics": r["metrics"],
                "run_id":  r["run_id"],
            }
            for r in all_results
        ],
    }
    with open(PROCESSED_DIR / "training_report.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Training report → %s", PROCESSED_DIR / "training_report.json")


if __name__ == "__main__":
    main()
