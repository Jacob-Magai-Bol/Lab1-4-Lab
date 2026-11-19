
"""Tests for the Lab 12 Iris classification API."""

import sys
from pathlib import Path

from fastapi.testclient import TestClient

LAB10_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB10_DIR))

from src.api.main import app, limiter

client = TestClient(app)

TEST_API_KEY = "test-iris-api-key"

VALID_INPUT = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}

AUTH_HEADERS = {"X-API-Key": TEST_API_KEY}


def setup_function():
    """Configure the test API key and reset rate-limit storage."""
    import os

    os.environ["IRIS_API_KEY"] = TEST_API_KEY
    limiter.reset()


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_predict_valid_input():
    response = client.post(
        "/predict",
        json=VALID_INPUT,
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    result = response.json()
    assert result["species"] == "setosa"
    assert result["prediction"] == 0
    assert result["model"] == "IrisClassifier"


def test_predict_invalid_input():
    invalid_input = VALID_INPUT.copy()
    invalid_input["sepal_length"] = -1

    response = client.post(
        "/predict",
        json=invalid_input,
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 422


def test_predict_missing_api_key():
    response = client.post("/predict", json=VALID_INPUT)

    assert response.status_code == 401


def test_predict_invalid_api_key():
    response = client.post(
        "/predict",
        json=VALID_INPUT,
        headers={"X-API-Key": "wrong-key"},
    )

    assert response.status_code == 401


def test_metrics_endpoint():
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "http_requests_total" in response.text
    assert "http_request_duration_seconds" in response.text
    assert "iris_predictions_total" in response.text


def test_rate_limit():
    responses = []

    for _ in range(11):
        response = client.post(
            "/predict",
            json=VALID_INPUT,
            headers=AUTH_HEADERS,
        )
        responses.append(response.status_code)

    assert responses.count(429) >= 1
