
"""Convert the trained sklearn pipeline to ONNX and compare ten test rows."""

from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "iris_pipeline.joblib"
ONNX_PATH = ROOT / "models" / "iris_pipeline.onnx"


def main():
    # Check that the trained Joblib model exists
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Run train_pipeline.py before convert_onnx.py"
        )

    # Load the trained Scikit-learn pipeline
    saved = joblib.load(MODEL_PATH)
    pipeline = saved["pipeline"]

    # Load Iris dataset
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    # Reproduce the same train-test split used during training
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Select ten genuine test samples
    X_ten = X_test.iloc[:10]
    X_ten_numpy = X_ten.to_numpy(dtype=np.float32)

    # Convert the Scikit-learn pipeline to ONNX
    initial_type = [
        ("float_input", FloatTensorType([None, 4]))
    ]

    onx = convert_sklearn(
        pipeline,
        initial_types=initial_type
    )

    # Save ONNX model
    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    ONNX_PATH.write_bytes(onx.SerializeToString())

    # Load ONNX model using ONNX Runtime
    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name

    # Generate ONNX predictions
    outputs = session.run(
        None,
        {input_name: X_ten_numpy}
    )

    onnx_labels = np.asarray(outputs[0]).reshape(-1).astype(int)

    # Generate Joblib predictions using the same ten samples
    joblib_labels = pipeline.predict(X_ten).astype(int)

    # Compare both model predictions
    agrees = np.array_equal(joblib_labels, onnx_labels)

    print("\n========== ONNX CONVERSION RESULTS ==========")
    print("Rows compared:", len(X_ten))
    print("Actual labels: ", y_test.iloc[:10].tolist())
    print("Joblib labels: ", joblib_labels.tolist())
    print("ONNX labels:   ", onnx_labels.tolist())
    print("All ten class predictions agree:", agrees)

    if not agrees:
        raise AssertionError(
            "Joblib and ONNX class predictions do not match."
        )

    print("\nONNX conversion and prediction comparison successful.")
    print(f"Saved ONNX model: {ONNX_PATH}")


if __name__ == "__main__":
    main()
