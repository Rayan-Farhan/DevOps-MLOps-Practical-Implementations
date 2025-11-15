from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_predict_missing_field_returns_422():
    payload = {
        # "Glucose": 120,  # intentionally omitted
        "Pregnancies": 2,
        "BloodPressure": 70,
        "SkinThickness": 20,
        "Insulin": 85,
        "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.45,
        "Age": 33,
    }

    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422


essential_payload = {
    "Pregnancies": 2,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 20,
    "Insulin": 85,
    "BMI": 28.5,
    "DiabetesPedigreeFunction": 0.45,
    "Age": 33,
}


def test_predict_invalid_negative_values_422():
    bad = {
        **essential_payload,
        "BMI": -1.0,  # invalid per schema ge=0
    }
    response = client.post("/api/predict", json=bad)
    assert response.status_code == 422


def test_predict_out_of_range_values_422():
    bad = {
        **essential_payload,
        "Age": 130,  # invalid per schema le=120
    }
    response = client.post("/api/predict", json=bad)
    assert response.status_code == 422
