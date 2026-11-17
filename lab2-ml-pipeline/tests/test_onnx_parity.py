"""Run with: python -m pytest lab2-ml-pipeline/tests -v"""
from pathlib import Path
import joblib
import numpy as np
import onnxruntime as ort
from sklearn.datasets import load_iris

ROOT = Path(__file__).resolve().parents[1]


def test_joblib_and_onnx_predictions_match_on_ten_rows():
    joblib_path = ROOT / "models" / "iris_pipeline.joblib"
    onnx_path = ROOT / "models" / "iris_pipeline.onnx"
    assert joblib_path.exists(), "Run train_pipeline.py first"
    assert onnx_path.exists(), "Run convert_onnx.py first"

    pipeline = joblib.load(joblib_path)["pipeline"]
    X = load_iris().data[:10].astype(np.float32)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    result = session.run(None, {session.get_inputs()[0].name: X})
    assert np.array_equal(pipeline.predict(X).astype(int), np.asarray(result[0]).reshape(-1).astype(int))
