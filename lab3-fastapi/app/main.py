"""FastAPI wrapper for the Lab 2 Iris pipeline."""
from pathlib import Path
import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(ROOT / "lab2-ml-pipeline" / "models" / "iris_pipeline.joblib")))
app = FastAPI(title="Iris Classifier API", version="1.0.0")

class PredictionRequest(BaseModel):
    features: list[float] = Field(
        ..., min_length=4, max_length=4,
        description="Exactly four values: sepal length, sepal width, petal length, petal width (cm)"
    )

    @field_validator("features")
    @classmethod
    def validate_finite_and_nonnegative(cls, values):
        if not all(np.isfinite(v) and v >= 0 for v in values):
            raise ValueError("Features must be finite, non-negative numbers.")
        return values

class PredictionResponse(BaseModel):
    predicted_class: int
    species: str

def load_artifact():
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. Run lab2-ml-pipeline/src/train_pipeline.py first."
        )
    return joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        artifact = load_artifact()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    model = artifact["pipeline"]
    target_names = artifact["target_names"]
    prediction = int(model.predict([request.features])[0])
    return {"predicted_class": prediction, "species": target_names[prediction]}
