import logging
import os
import pickle
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException

from app.config import settings
from app.schemas.diabetes_schema import DiabetesInput
from app.utils.metrics import inc_prediction

router = APIRouter()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Metrics are defined centrally in app.utils.metrics

# SQLite path for prediction logging (used by drift detection)
_PREDICTIONS_DB = (
    Path(__file__).resolve().parent.parent.parent / "data" / "predictions.db"
)


def _log_prediction(data, result: str) -> None:
    """Persist each prediction to SQLite for downstream drift detection."""
    try:
        _PREDICTIONS_DB.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(_PREDICTIONS_DB)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id                        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp                 TEXT,
                pregnancies               REAL,
                glucose                   REAL,
                blood_pressure            REAL,
                skin_thickness            REAL,
                insulin                   REAL,
                bmi                       REAL,
                diabetes_pedigree_function REAL,
                age                       REAL,
                result                    TEXT
            )
            """
        )
        conn.execute(
            """INSERT INTO predictions
               (timestamp, pregnancies, glucose, blood_pressure, skin_thickness,
                insulin, bmi, diabetes_pedigree_function, age, result)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                data.Pregnancies,
                data.Glucose,
                data.BloodPressure,
                data.SkinThickness,
                data.Insulin,
                data.BMI,
                data.DiabetesPedigreeFunction,
                data.Age,
                result,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        # Prediction logging must never break the API response
        logger.warning("Prediction logging failed (non-fatal): %s", e)


# Load ML model at module import time
model_path = settings.MODEL_PATH
model = None
scaler = None
model_features = None
train_medians = {}
zero_impute_cols = []

if not os.path.exists(model_path):
    logger.error(f"Model file not found at path: {model_path}")
else:
    logger.info(f"Model file found at: {model_path}")
    try:
        with open(model_path, "rb") as f:
            loaded = pickle.load(f)
            model = loaded.get("model")
            scaler = loaded.get("scaler")
            model_features = loaded.get("features")
            train_medians = loaded.get("train_medians", {})
            zero_impute_cols = loaded.get("zero_impute_cols", [])
            logger.info(
                "Model bundle loaded successfully. "
                "Model type: %s, scaler: %s, features: %s",
                type(model),
                type(scaler),
                model_features,
            )
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e} ({type(e).__name__})")
        model = None
        scaler = None
        model_features = None


@router.get("/health")
def health_check():
    return {"status": "ok", "message": "Diabetes Prediction API is running!"}


@router.post("/predict")
def predict_diabetes(data: DiabetesInput):
    if model is None:
        logger.error("Model not loaded.")
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Use model's feature order if available, otherwise use default order
        if model_features:
            row = [getattr(data, feature) for feature in model_features]
            input_data = np.array([row], dtype=float)
        else:
            input_data = np.array(
                [
                    [
                        data.Pregnancies,
                        data.Glucose,
                        data.BloodPressure,
                        data.SkinThickness,
                        data.Insulin,
                        data.BMI,
                        data.DiabetesPedigreeFunction,
                        data.Age,
                    ]
                ],
                dtype=float,
            )

        # Apply the same zero-to-median imputation used during training.
        # In the Pima dataset, 0 is a sentinel for missing in biological columns.
        if train_medians and zero_impute_cols:
            features = model_features or [
                "Pregnancies",
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "Insulin",
                "BMI",
                "DiabetesPedigreeFunction",
                "Age",
            ]
            for i, feat in enumerate(features):
                if feat in zero_impute_cols and input_data[0, i] == 0.0:
                    input_data[0, i] = train_medians.get(feat, 0.0)

        # Apply feature scaling
        if scaler is not None:
            try:
                input_data = scaler.transform(input_data)
            except Exception as se:
                logger.warning(f"Scaler transform failed: {se}; continuing with raw inputs")

        prediction = model.predict(input_data)
        result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
        inc_prediction(result)
        _log_prediction(data, result)

        logger.info("Prediction made successfully: %s", result)
        return {"prediction": int(prediction[0]), "result": result}

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")
