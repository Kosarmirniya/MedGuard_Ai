from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_risk_assessment():
    payload = {
        "HighBP": 1,
        "HighChol": 1,
        "CholCheck": 1,
        "BMI": 28,
        "Smoker": 0,
        "Stroke": 0,
        "HeartDiseaseorAttack": 0,
        "PhysActivity": 1,
        "Fruits": 1,
        "Veggies": 1,
        "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1,
        "NoDocbcCost": 0,
        "GenHlth": 3,
        "MentHlth": 2,
        "PhysHlth": 3,
        "DiffWalk": 0,
        "Sex": 1,
        "Age": 7,
        "Education": 4,
        "Income": 5
    }

    response = client.post("/risk-assessment", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "result" in data
    assert "prediction" in data["result"]
    assert "probabilities" in data["result"]

def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"

def test_invalid_risk_request():
    payload = {}

    response = client.post("/risk-assessment", json=payload)

    assert response.status_code == 422
    