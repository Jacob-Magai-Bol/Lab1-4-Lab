import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import main

client = TestClient(main.app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_valid_question(monkeypatch):
    def mock_answer_question(question):
        return {
            "answer": "The probation period is six months.",
            "sources": [{"page": 8}]
        }

    monkeypatch.setattr(main, "answer_question", mock_answer_question)

    response = client.post(
        "/ask",
        json={"question": "What is the probation period?"}
    )

    assert response.status_code == 200
    assert "six months" in response.json()["answer"]
    assert response.json()["sources"][0]["page"] == 8


def test_question_too_short():
    response = client.post(
        "/ask",
        json={"question": "Hi"}
    )

    assert response.status_code == 422
