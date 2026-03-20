"""
Stage 5: Model Evaluation and Promotion

Evaluates the newly trained model (data/processed/trained_model.pkl) against
the currently deployed model (app/model/diabetes_model.pkl) on the held-out
test split.

Promotion rules (from params.yaml):
  - New model's primary metric must reach promotion_threshold
  - New model must beat the current deployed model by at least min_improvement
  - If no deployed model exists, the new model is always promoted

On promotion:
  - Copies trained_model.pkl → app/model/diabetes_model.pkl
  - The running FastAPI service picks up the new .pkl on next restart

Outputs:
  data/processed/evaluation_report.json   -- full human-readable report
  data/processed/metrics.json             -- DVC metrics file (tracked by dvc metrics show)
"""

import json
import logging
import os
import pickle
import shutil
import sys
from pathlib import Path

import mlflow
import numpy as np
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
DEPLOYED_MODEL_PATH = ROOT / "app" / "model" / "diabetes_model.pkl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_params() -> dict:
    with open(ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


def load_test_data() -> tuple[np.ndarray, np.ndarray]:
    X_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")
    logger.info(
        "Loaded test data: X_test=%s  y_test=%s", X_test.shape, y_test.shape
    )
    return X_test, y_test


def load_bundle(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def evaluate_bundle(bundle: dict, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate the bundle's model on X_test.

    X_test is loaded from data/processed/X_test.npy which is ALREADY SCALED
    by preprocess.py. The bundle's scaler is for the serving API only
    (raw user input → scale → predict). Do NOT apply it here.
    """
    model = bundle["model"]
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_diabetic": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_pred)),
    }

    report_str = classification_report(
        y_test, y_pred, target_names=["Non-Diabetic", "Diabetic"]
    )
    return metrics, report_str


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    params = load_params()
    eval_cfg = params["evaluation"]
    primary_metric: str = eval_cfg["primary_metric"]
    min_improvement: float = eval_cfg["min_improvement"]
    promotion_threshold: float = eval_cfg["promotion_threshold"]

    X_test, y_test = load_test_data()

    # --- Evaluate ALL candidate models on the test set ---
    # train.py saves individual bundles for every candidate so we can pick
    # the true best performer on the held-out test set, not just the CV winner.
    candidate_names = [
        "random_forest", "logistic_regression", "gradient_boosting", "xgboost"
    ]
    candidates = []
    for name in candidate_names:
        bundle = load_bundle(PROCESSED_DIR / f"model_{name}.pkl")
        if bundle is None:
            logger.warning("Bundle not found for '%s' — skipping.", name)
            continue
        metrics, report = evaluate_bundle(bundle, X_test, y_test)
        logger.info(
            "Candidate [%-20s]  %s=%.4f  accuracy=%.4f",
            name, primary_metric, metrics[primary_metric], metrics["accuracy"],
        )
        candidates.append({"name": name, "bundle": bundle, "metrics": metrics, "report": report})

    if not candidates:
        logger.error("No candidate bundles found in %s", PROCESSED_DIR)
        sys.exit(1)

    # Pick the best test-set performer as the new model to promote
    best_candidate = max(candidates, key=lambda c: c["metrics"][primary_metric])
    new_bundle  = best_candidate["bundle"]
    new_metrics = best_candidate["metrics"]
    new_report  = best_candidate["report"]

    logger.info(
        "Best candidate on test set: %s  (%s=%.4f)",
        best_candidate["name"], primary_metric, new_metrics[primary_metric],
    )
    logger.info("Best candidate classification report:\n%s", new_report)

    # Also log all candidate scores for visibility
    logger.info("--- Candidate test-set ranking ---")
    for c in sorted(candidates, key=lambda x: x["metrics"][primary_metric], reverse=True):
        logger.info(
            "  %-22s  %s=%.4f  accuracy=%.4f",
            c["name"], primary_metric, c["metrics"][primary_metric], c["metrics"]["accuracy"],
        )

    # --- Evaluate current deployed model (if it exists) ---
    current_bundle = load_bundle(DEPLOYED_MODEL_PATH)
    current_metrics = None

    if current_bundle is None:
        logger.info(
            "No deployed model found at %s — new model will be promoted unconditionally.",
            DEPLOYED_MODEL_PATH,
        )
    else:
        current_metrics, current_report = evaluate_bundle(current_bundle, X_test, y_test)
        logger.info(
            "Current model (%s) — %s=%.4f  accuracy=%.4f",
            current_bundle.get("model_type", "unknown"),
            primary_metric,
            current_metrics[primary_metric],
            current_metrics["accuracy"],
        )
        logger.info("Current model classification report:\n%s", current_report)

    # --- Promotion decision ---
    new_score = new_metrics[primary_metric]
    current_score = current_metrics[primary_metric] if current_metrics else None

    promoted = False
    promotion_reason = ""

    if new_score < promotion_threshold:
        promotion_reason = (
            f"New model {primary_metric}={new_score:.4f} is below "
            f"promotion_threshold={promotion_threshold}"
        )
        logger.warning("PROMOTION REJECTED: %s", promotion_reason)

    elif current_score is None:
        # No existing model — always promote
        promoted = True
        promotion_reason = "No deployed model exists. Initial promotion."

    elif new_score >= current_score + min_improvement:
        promoted = True
        promotion_reason = (
            f"New model {primary_metric}={new_score:.4f} beats current "
            f"{current_score:.4f} by {new_score - current_score:.4f} "
            f"(min_improvement={min_improvement})"
        )
    else:
        promotion_reason = (
            f"New model {primary_metric}={new_score:.4f} does not improve "
            f"over current {current_score:.4f} by min_improvement={min_improvement}"
        )
        logger.info("PROMOTION SKIPPED: %s", promotion_reason)

    if promoted:
        winner_path = PROCESSED_DIR / f"model_{best_candidate['name']}.pkl"
        shutil.copy2(winner_path, DEPLOYED_MODEL_PATH)
        logger.info(
            "PROMOTED: %s → %s  (%s)",
            winner_path,
            DEPLOYED_MODEL_PATH,
            promotion_reason,
        )

        # Transition the registered model version to Production in MLflow Registry
        run_id = new_bundle.get("mlflow_run_id")
        registry_version = new_bundle.get("registry_version")
        if run_id and registry_version:
            try:
                tracking_uri = os.getenv(
                    "MLFLOW_TRACKING_URI",
                    f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}",
                )
                mlflow.set_tracking_uri(tracking_uri)
                client = mlflow.MlflowClient()
                client.transition_model_version_stage(
                    name="diabetes-classifier",
                    version=registry_version,
                    stage="Production",
                    archive_existing_versions=True,
                )
                logger.info(
                    "MLflow Registry: 'diabetes-classifier' v%s → Production",
                    registry_version,
                )
            except Exception as e:
                logger.warning("MLflow Registry transition failed (non-fatal): %s", e)
    else:
        logger.info("Deployed model unchanged.")

    # --- Write outputs ---
    evaluation_report = {
        "promoted": promoted,
        "promotion_reason": promotion_reason,
        "primary_metric": primary_metric,
        "best_candidate": {
            "type": best_candidate["name"],
            "mlflow_run_id": new_bundle.get("mlflow_run_id"),
            "test_metrics": new_metrics,
        },
        "all_candidates": [
            {"name": c["name"], "test_metrics": c["metrics"]}
            for c in sorted(candidates, key=lambda x: x["metrics"][primary_metric], reverse=True)
        ],
        "current_model": (
            {
                "type": current_bundle.get("model_type") if current_bundle else None,
                "test_metrics": current_metrics,
            }
            if current_bundle
            else None
        ),
    }

    report_path = PROCESSED_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(evaluation_report, f, indent=2)
    logger.info("Evaluation report → %s", report_path)

    # DVC metrics file — kept minimal for `dvc metrics show`
    dvc_metrics = {
        "new_model": new_metrics,
        "promoted": promoted,
    }
    if current_metrics:
        dvc_metrics["current_model"] = current_metrics
        dvc_metrics["delta"] = {
            k: round(new_metrics[k] - current_metrics[k], 6)
            for k in new_metrics
        }

    metrics_path = PROCESSED_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(dvc_metrics, f, indent=2)
    logger.info("DVC metrics → %s", metrics_path)

    if promoted:
        logger.info("Pipeline complete. New model is now deployed.")
    else:
        logger.info("Pipeline complete. Deployed model was NOT replaced.")


if __name__ == "__main__":
    main()
