from fastapi import APIRouter, HTTPException
import numpy as np
import pickle
import logging
import os
from app.schemas.diabetes_schema import DiabetesInput
from app.config import settings
from prometheus_client import Counter

router = APIRouter()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Metrics: count successful predictions
PREDICTION_COUNTER = Counter(
    "diabetes_predictions_total",
    "Total number of predictions made",
    ["result"],
)

# Load ML model at module import time
model_path = settings.MODEL_PATH
model = None
scaler = None
model_features = None

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
            logger.info(
                f"Model bundle loaded successfully. Model type: {type(model)}, scaler: {type(scaler)}, features: {model_features}"
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
            input_data = np.array([row])
        else:
            input_data = np.array([[
                data.Pregnancies,
                data.Glucose,
                data.BloodPressure,
                data.SkinThickness,
                data.Insulin,
                data.BMI,
                data.DiabetesPedigreeFunction,
                data.Age
            ]])

        # Apply feature scaling if scaler was saved with the model
        if scaler is not None:
            try:
                input_data = scaler.transform(input_data)
            except Exception as se:
                logger.warning(f"Scaler transform failed: {se}; continuing with raw inputs")

        prediction = model.predict(input_data)
        result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
        try:
            PREDICTION_COUNTER.labels(result=result).inc()
        except Exception:
            # Metrics failure should not break API
            pass

        logger.info(f"✅ Prediction made successfully: {result}")
        return {"prediction": int(prediction[0]), "result": result}

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")