import pickle
import numpy as np
import os

MODEL_PATH = "app/model/diabetes_model.pkl"

def test_model_file_exists():
    assert os.path.exists(MODEL_PATH), "Model file not found!"

def test_model_prediction_shape():
    with open(MODEL_PATH, "rb") as f:
        loaded = pickle.load(f)
        if isinstance(loaded, dict) and 'model' in loaded:
            model = loaded['model']
        else:
            model = loaded

    sample_input = np.array([[2, 120, 70, 20, 85, 28.5, 0.45, 33]])
    prediction = model.predict(sample_input)
    
    assert prediction.shape == (1,), "Prediction output shape mismatch"
    assert prediction[0] in [0, 1], "Invalid prediction value"