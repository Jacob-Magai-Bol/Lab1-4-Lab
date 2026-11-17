"""API tests. Generate the Lab 2 model before running pytest."""
from pathlib import Path
import subprocess
import sys
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lab3-fastapi"))
from app.main import app

client = TestClient(app)


def ensure_model():
    model = ROOT / "lab2-ml-pipeline" / "models" / "iris_pipeline.joblib"
    if not model.exists():
        subprocess.run(
            [sys.executable, str(ROOT / "lab2-ml-pipeline" / "src" / "train_pipeline.py")],
            check=True
        )


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_valid_iris_record():
    ensure_model()
    response = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert response.status_code == 200
    assert response.json()["species"] in {"setosa", "versicolor", "virginica"}


def test_predict_rejects_wrong_number_of_features():
    response = client.post("/predict", json={"features": [5.1, 3.5, 1.4]})
    assert response.status_code == 422
