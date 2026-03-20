"""
Stage 6: Data Drift Detection (Evidently AI)

Compares the feature distributions of recent live predictions (current dataset)
against the original training data (reference dataset) using Evidently AI.

Run standalone at any time, or automatically via the scheduled retrain workflow:
  python pipeline/drift.py

Exit codes:
  0 — No drift detected (or insufficient data to run analysis)
  2 — Drift detected → the retrain workflow uses this to trigger dvc repro

Outputs:
  data/processed/drift_report.json
"""

import json
import logging
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import yaml
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.report import Report

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("drift")

FEATURE_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]

# Column name mapping: predictions DB → training feature names
DB_TO_FEATURE = {
    "pregnancies":               "Pregnancies",
    "glucose":                   "Glucose",
    "blood_pressure":            "BloodPressure",
    "skin_thickness":            "SkinThickness",
    "insulin":                   "Insulin",
    "bmi":                       "BMI",
    "diabetes_pedigree_function": "DiabetesPedigreeFunction",
    "age":                       "Age",
}


def load_params() -> dict:
    with open(ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


def load_reference() -> pd.DataFrame:
    """Training CSV as the reference distribution."""
    csv_path = RAW_DIR / "diabetes.csv"
    if not csv_path.exists():
        logger.error("Reference CSV not found: %s", csv_path)
        sys.exit(1)
    df = pd.read_csv(csv_path)
    return df[FEATURE_COLUMNS]


def load_current(db_path: Path, min_rows: int) -> pd.DataFrame | None:
    """Recent live predictions from the SQLite prediction log."""
    if not db_path.exists():
        logger.warning("Predictions DB not found: %s", db_path)
        return None

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(
            "SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 1000", conn
        )
    except Exception as e:
        logger.warning("Could not read predictions table: %s", e)
        conn.close()
        return None
    conn.close()

    if len(df) < min_rows:
        logger.warning(
            "Only %d prediction rows available (need ≥%d). Skipping drift analysis.",
            len(df), min_rows,
        )
        return None

    df = df.rename(columns=DB_TO_FEATURE)
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    return df[available]


def write_report(path: Path, payload: dict) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("Drift report → %s", path)


def main() -> None:
    params     = load_params()
    drift_cfg  = params.get("drift", {})
    min_rows   = drift_cfg.get("min_prediction_rows", 50)
    db_path    = ROOT / drift_cfg.get("predictions_db", "data/predictions.db")
    report_path = PROCESSED_DIR / "drift_report.json"

    reference = load_reference()
    current   = load_current(db_path, min_rows)

    if current is None:
        write_report(report_path, {
            "status":       "skipped",
            "reason":       "insufficient_prediction_data",
            "min_required": min_rows,
        })
        logger.info("Drift check skipped — not enough live prediction data yet.")
        sys.exit(0)

    logger.info(
        "Running Evidently drift analysis: reference=%d rows, current=%d rows",
        len(reference), len(current),
    )

    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=reference, current_data=current)
    result = report.as_dict()

    # Extract top-level drift flag from Evidently result structure
    drift_detected = False
    try:
        for metric in result.get("metrics", []):
            if metric.get("metric") == "DatasetDriftMetric":
                drift_detected = metric["result"].get("dataset_drift", False)
                break
    except (KeyError, TypeError):
        pass

    # Per-feature drift summary
    feature_drift = {}
    try:
        for metric in result.get("metrics", []):
            if metric.get("metric") == "ColumnDriftMetric":
                col  = metric["result"]["column_name"]
                drifted = metric["result"].get("drift_detected", False)
                score   = metric["result"].get("drift_score", None)
                feature_drift[col] = {"drift_detected": drifted, "drift_score": score}
    except (KeyError, TypeError):
        pass

    summary = {
        "status":           "completed",
        "drift_detected":   drift_detected,
        "reference_rows":   len(reference),
        "current_rows":     len(current),
        "feature_drift":    feature_drift,
        "full_report":      result,
    }
    write_report(report_path, summary)

    drifted_features = [k for k, v in feature_drift.items() if v.get("drift_detected")]
    if drift_detected:
        logger.warning(
            "DATA DRIFT DETECTED in %d feature(s): %s",
            len(drifted_features), drifted_features,
        )
        logger.warning("Consider triggering a retraining run (dvc repro).")
        sys.exit(2)
    else:
        logger.info(
            "No significant data drift detected. All %d features stable.", len(reference.columns)
        )
        sys.exit(0)


if __name__ == "__main__":
    main()
