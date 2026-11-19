"""Lab 11 FastAPI service with Prometheus monitoring."""
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "iris_pipeline.joblib"

artifact = joblib.load(MODEL_PATH)
pipeline = artifact["pipeline"]
feature_names = artifact["feature_names"]
target_names = artifact["target_names"]

app = FastAPI(
    title="Lab 11 Iris Classifier API",
    description="Iris classification API with Prometheus monitoring.",
    version="1.1.0",
)

# Count HTTP requests by method, endpoint and status code.
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "http_status"],
)

# Measure HTTP request latency in seconds.
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
)

# Count successful predictions by predicted species.
PREDICTION_COUNT = Counter(
    "iris_predictions_total",
    "Total number of successful Iris predictions",
    ["species"],
)


@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = perf_counter()

    response = await call_next(request)

    # Exclude Prometheus scrapes to avoid counting monitoring traffic.
    if request.url.path != "/metrics":
        elapsed = perf_counter() - start_time
        endpoint = request.url.path

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            http_status=str(response.status_code),
        ).inc()

        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=endpoint,
        ).observe(elapsed)

    return response


class IrisInput(BaseModel):
    sepal_length: float = Field(gt=0, description="Sepal length in cm")
    sepal_width: float = Field(gt=0, description="Sepal width in cm")
    petal_length: float = Field(gt=0, description="Petal length in cm")
    petal_width: float = Field(gt=0, description="Petal width in cm")


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/predict")
def predict(data: IrisInput):
    values = [
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width,
    ]

    if not np.all(np.isfinite(values)):
        raise HTTPException(status_code=422, detail="Input values must be finite.")

    prediction = int(pipeline.predict([values])[0])
    species = target_names[prediction]

    PREDICTION_COUNT.labels(species=species).inc()

    return {
        "prediction": prediction,
        "species": species,
        "model": "IrisClassifier",
    }
