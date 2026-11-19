"""Lab 10 FastAPI service for the registered Iris classification project."""
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "iris_pipeline.joblib"

artifact = joblib.load(MODEL_PATH)
pipeline = artifact["pipeline"]
feature_names = artifact["feature_names"]
target_names = artifact["target_names"]

app = FastAPI(
    title="Lab 10 Iris Classifier API",
    description="Iris classification API using an MLflow-tracked model.",
    version="1.0.0",
)


class IrisInput(BaseModel):
    sepal_length: float = Field(gt=0, description="Sepal length in cm")
    sepal_width: float = Field(gt=0, description="Sepal width in cm")
    petal_length: float = Field(gt=0, description="Petal length in cm")
    petal_width: float = Field(gt=0, description="Petal width in cm")


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


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

    return {
        "prediction": prediction,
        "species": target_names[prediction],
        "model": "IrisClassifier",
    }
