"""Train, evaluate and save a simple Iris classifier (Lab 1)."""
from pathlib import Path
import json
import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    model_path = MODEL_DIR / "iris_classifier.joblib"
    joblib.dump(
        {"model": model, "feature_names": list(X.columns),
         "target_names": list(iris.target_names)},
        model_path
    )

    report = classification_report(
        y_test, predictions, target_names=iris.target_names, output_dict=True
    )
    results = {"accuracy": accuracy, "model_path": str(model_path),
               "classification_report": report}
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {OUTPUT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
