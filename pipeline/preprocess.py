"""
Stage 3: Preprocessing

Reads data/raw/diabetes.csv, handles missing values, splits into train/test,
fits a StandardScaler on the training set, and serialises all outputs.

The Pima Indians Diabetes dataset uses 0 as a placeholder for missing values
in several columns (Glucose, BloodPressure, SkinThickness, Insulin, BMI).
These are replaced with column medians (computed from the training split only
to prevent data leakage).

Outputs:
  data/processed/X_train.npy
  data/processed/X_test.npy
  data/processed/y_train.npy
  data/processed/y_test.npy
  data/processed/scaler.pkl          -- fitted StandardScaler
  data/processed/feature_names.json  -- ordered list of feature names
"""

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("preprocess")


def load_params() -> dict:
    with open(ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


def replace_zeros_with_nan(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            zeros = (df[col] == 0).sum()
            if zeros:
                logger.info(
                    "Replacing %d zero values in '%s' with NaN", zeros, col
                )
                df.loc[df[col] == 0, col] = np.nan
    return df


def impute_with_train_medians(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Impute NaN with medians computed solely from the training split."""
    train_medians = X_train.median()
    X_train = X_train.fillna(train_medians)
    X_test = X_test.fillna(train_medians)
    logger.info("Imputed missing values using training-set medians")
    return X_train, X_test, train_medians


def main() -> None:
    params = load_params()
    data_cfg = params["data"]

    csv_path = RAW_DIR / "diabetes.csv"
    if not csv_path.exists():
        logger.error("Raw CSV not found: %s", csv_path)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    logger.info("Loaded %s: %d rows × %d columns", csv_path, *df.shape)

    # Drop any columns beyond the expected feature + label set
    expected = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
    ]
    df = df[[c for c in expected if c in df.columns]]

    # Separate features and label
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    feature_names = list(X.columns)

    # Replace biologically impossible zeros with NaN
    zero_cols = data_cfg.get("replace_zeros_with_nan", [])
    X = replace_zeros_with_nan(X, zero_cols)

    # Train / test split (stratified to preserve class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=data_cfg["test_size"],
        random_state=data_cfg["random_state"],
        stratify=y,
    )
    logger.info(
        "Split: %d train rows, %d test rows (%.0f / %.0f)",
        len(X_train),
        len(X_test),
        (1 - data_cfg["test_size"]) * 100,
        data_cfg["test_size"] * 100,
    )

    # Impute missing values (leakage-safe: fit on train only)
    X_train, X_test, train_medians = impute_with_train_medians(X_train, X_test)

    # Scale: fit ONLY on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.info("StandardScaler fitted on training set")

    # Persist outputs
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    np.save(PROCESSED_DIR / "X_train.npy", X_train_scaled)
    np.save(PROCESSED_DIR / "X_test.npy", X_test_scaled)
    np.save(PROCESSED_DIR / "y_train.npy", y_train.to_numpy())
    np.save(PROCESSED_DIR / "y_test.npy", y_test.to_numpy())

    with open(PROCESSED_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(PROCESSED_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    # Save training medians so inference can apply the same imputation
    medians_dict = train_medians.to_dict()
    with open(PROCESSED_DIR / "train_medians.json", "w") as f:
        json.dump(medians_dict, f, indent=2)

    logger.info(
        "Saved processed arrays to %s  (features: %s)",
        PROCESSED_DIR,
        feature_names,
    )


if __name__ == "__main__":
    main()
