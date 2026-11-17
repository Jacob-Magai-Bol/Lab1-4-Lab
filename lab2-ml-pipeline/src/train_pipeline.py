"""Train a reusable preprocessing + classifier pipeline (Lab 2)."""
from pathlib import Path
import json
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    joblib.dump(
        {"pipeline": pipeline, "feature_names": list(X.columns),
         "target_names": list(iris.target_names)},
        MODEL_DIR / "iris_pipeline.joblib"
    )
    metrics = {
        "accuracy": accuracy,
        "classification_report": classification_report(
            y_test, predictions, target_names=iris.target_names, output_dict=True
        )
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Pipeline test accuracy: {accuracy:.4f}")
    print(f"Saved: {MODEL_DIR / 'iris_pipeline.joblib'}")


if __name__ == "__main__":
    main()
