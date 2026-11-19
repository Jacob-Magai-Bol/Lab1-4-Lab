
"""Lab 12 Iris API with authentication, rate limiting, and Prometheus monitoring."""

import os
import secrets
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import Response
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "iris_pipeline.joblib"

artifact = joblib.load(MODEL_PATH)
pipeline = artifact["pipeline"]
feature_names = artifact["feature_names"]
target_names = artifact["target_names"]

# Rate limiting: 10 prediction requests per minute per IP address.
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Lab 12 Iris Classifier API",
    description="Iris classification API with authentication, rate limiting, and Prometheus monitoring.",
    version="1.2.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# API key must be configured through the IRIS_API_KEY environment variable.
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str | None = Depends(api_key_header)):
    expected_api_key = os.getenv("IRIS_API_KEY")

    if not expected_api_key:
        raise HTTPException(
            status_code=503,
            detail="API key is not configured.",
        )

    if api_key is None or not secrets.compare_digest(
        api_key, expected_api_key
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
        )

    return api_key


# Prometheus HTTP request counter.
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "http_status"],
)

# Prometheus HTTP request latency.
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
)

# Prometheus successful prediction counter.
PREDICTION_COUNT = Counter(
    "iris_predictions_total",
    "Total number of successful Iris predictions",
    ["species"],
)


@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = perf_counter()
    response = await call_next(request)

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
@limiter.limit("10/minute")
def predict(
    request: Request,
    data: IrisInput,
    api_key: str = Depends(verify_api_key),
):
    values = [
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width,
    ]

    if not np.all(np.isfinite(values)):
        raise HTTPException(
            status_code=422,
            detail="Input values must be finite.",
        )

    prediction = int(pipeline.predict([values])[0])
    species = target_names[prediction]

    PREDICTION_COUNT.labels(species=species).inc()

    return {
        "prediction": prediction,
        "species": species,
        "model": "IrisClassifier",
    }
