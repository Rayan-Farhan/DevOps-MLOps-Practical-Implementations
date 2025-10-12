from fastapi import APIRouter, HTTPException
import numpy as np
import pickle
import logging
from app.schemas.diabetes_schema import DiabetesInput
import os

router = APIRouter()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

model_path = os.path.join(os.path.dirname(__file__), "..", "model", "diabetes_model.pkl")

model = None
scaler = None
model_features = None
try:
    with open(model_path, "rb") as f:
        loaded = pickle.load(f)
        model = loaded.get("model")
        scaler = loaded.get("scaler")
        model_features = loaded.get("features")
        logger.info(f"Model bundle loaded. Model type: {type(model)}, scaler: {type(scaler)}, features: {model_features}")

except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None
    scaler = None
    model_features = None


@router.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "Diabetes Prediction API is running!"}


@router.post("/predict")
def predict_diabetes(data: DiabetesInput):
    """Predict diabetes likelihood based on input"""
    if model is None:
        logger.error("Model not loaded.")
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
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

        # apply scaler if present
        if scaler is not None:
            try:
                input_data = scaler.transform(input_data)
            except Exception as se:
                logger.warning(f"Scaler transform failed: {se}; continuing with raw inputs")

        prediction = model.predict(input_data)
        result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"

        logger.info(f"✅ Prediction made successfully: {result}")
        return {"prediction": int(prediction[0]), "result": result}

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")