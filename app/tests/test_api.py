from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_diabetes():
    payload = {
        "Pregnancies": 2,
        "Glucose": 120,
        "BloodPressure": 70,
        "SkinThickness": 20,
        "Insulin": 85,
        "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.45,
        "Age": 33
    }

    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    assert "prediction" in json_data
    assert "result" in json_data
    assert json_data["prediction"] in [0, 1]