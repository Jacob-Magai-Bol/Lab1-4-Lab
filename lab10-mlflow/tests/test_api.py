"""Tests for the Lab 10 Iris classification API."""
import sys
from pathlib import Path

from fastapi.testclient import TestClient

# Add the Lab 10 project directory to Python's import path.
LAB10_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB10_DIR))

from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_predict_valid_input():
    response = client.post(
        "/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert result["species"] == "setosa"
    assert result["prediction"] == 0
    assert result["model"] == "IrisClassifier"


def test_predict_invalid_input():
    response = client.post(
        "/predict",
        json={
            "sepal_length": -1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        },
    )

    assert response.status_code == 422
