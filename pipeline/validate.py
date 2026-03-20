"""
Stage 2: Data Validation

Validates data/raw/diabetes.csv against expected schema and statistical constraints.

Checks performed:
  1. Required columns are present
  2. No unexpected additional columns
  3. Row count >= params.data.min_rows
  4. No fully-null columns
  5. Per-column value ranges (using the same bounds as the Pydantic schema)
  6. Outcome class balance within [class_balance_min, class_balance_max]

Exits with code 1 on any hard failure so the DVC pipeline stops early.

Outputs:
  data/raw/validation_report.json
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("validate")

# Column-level value bounds (mirrors diabetes_schema.py)
COLUMN_BOUNDS = {
    "Pregnancies": (0, 20),
    "Glucose": (0, 300),
    "BloodPressure": (0, 200),
    "SkinThickness": (0, 100),
    "Insulin": (0, 900),
    "BMI": (0.0, 80.0),
    "DiabetesPedigreeFunction": (0.0, 3.0),
    "Age": (0, 120),
    "Outcome": (0, 1),
}
REQUIRED_COLUMNS = list(COLUMN_BOUNDS.keys())


def load_params() -> dict:
    with open(ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


def validate(df: pd.DataFrame, params: dict) -> tuple[bool, list[str], list[str]]:
    """
    Returns (passed, errors, warnings).
    errors   → hard failures that block the pipeline
    warnings → soft issues logged but not blocking
    """
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    # 2. Unexpected columns
    extra_cols = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    if extra_cols:
        warnings.append(f"Unexpected extra columns (will be ignored): {extra_cols}")

    # Stop early if columns are missing — remaining checks would crash
    if missing_cols:
        return False, errors, warnings

    # 3. Row count
    min_rows = params["data"]["min_rows"]
    if len(df) < min_rows:
        errors.append(f"Row count {len(df)} is below minimum {min_rows}")
    else:
        logger.info("Row count check passed: %d rows", len(df))

    # 4. Fully-null columns
    null_cols = df[REQUIRED_COLUMNS].columns[df[REQUIRED_COLUMNS].isnull().all()].tolist()
    if null_cols:
        errors.append(f"Columns are entirely null: {null_cols}")

    # 5. Per-column value ranges (applied to non-null values only)
    for col, (lo, hi) in COLUMN_BOUNDS.items():
        if col not in df.columns:
            continue
        series = df[col].dropna()
        out_of_range = ((series < lo) | (series > hi)).sum()
        if out_of_range > 0:
            pct = out_of_range / len(series) * 100
            msg = (
                f"Column '{col}': {out_of_range} values ({pct:.1f}%) "
                f"outside [{lo}, {hi}]"
            )
            if pct > 5:
                errors.append(msg)
            else:
                warnings.append(msg)
        else:
            logger.info("Range check passed: %s in [%s, %s]", col, lo, hi)

    # 6. Class balance
    if "Outcome" in df.columns:
        positive_rate = df["Outcome"].mean()
        lo = params["data"]["class_balance_min"]
        hi = params["data"]["class_balance_max"]
        if not (lo <= positive_rate <= hi):
            warnings.append(
                f"Class imbalance: positive rate={positive_rate:.3f} outside [{lo}, {hi}]"
            )
        else:
            logger.info("Class balance check passed: positive rate=%.3f", positive_rate)

    passed = len(errors) == 0
    return passed, errors, warnings


def main() -> None:
    params = load_params()
    csv_path = RAW_DIR / "diabetes.csv"

    if not csv_path.exists():
        logger.error("Raw CSV not found: %s — run ingest stage first.", csv_path)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    logger.info("Loaded %s: %d rows × %d columns", csv_path, *df.shape)

    passed, errors, warnings = validate(df, params)

    for w in warnings:
        logger.warning(w)
    for e in errors:
        logger.error(e)

    report = {
        "passed": passed,
        "rows": len(df),
        "columns": list(df.columns),
        "errors": errors,
        "warnings": warnings,
        "stats": {
            col: {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "null_count": int(df[col].isnull().sum()),
            }
            for col in REQUIRED_COLUMNS
            if col in df.columns
        },
    }

    report_path = RAW_DIR / "validation_report.json"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    if passed:
        logger.info("Validation PASSED — report written to %s", report_path)
    else:
        logger.error("Validation FAILED — see errors above. Pipeline stopped.")
        sys.exit(1)


if __name__ == "__main__":
    main()
