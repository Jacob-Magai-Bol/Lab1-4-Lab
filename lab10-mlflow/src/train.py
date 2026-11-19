"""Lab 10: Train Iris classifier, track with MLflow, and register model."""
from pathlib import Path
import json

import mlflow
import mlflow.sklearn
import joblib
from mlflow.models import infer_signature
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
TRACKING_DB = ROOT / "mlflow.db"

MIN_ACCURACY = 0.90
EXPERIMENT_NAME = "lab10-iris-classification"
REGISTERED_MODEL_NAME = "IrisClassifier"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB.as_posix()}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    with mlflow.start_run() as run:
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        accuracy = float(accuracy_score(y_test, predictions))

        mlflow.log_params({
            "model": "LogisticRegression",
            "max_iter": 1000,
            "random_state": 42,
            "test_size": 0.2,
            "stratify": True,
            "features": len(iris.feature_names),
            "training_samples": len(X_train),
            "testing_samples": len(X_test),
        })

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("minimum_required_accuracy", MIN_ACCURACY)

        signature = infer_signature(X_train, pipeline.predict(X_train))

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="iris_pipeline",
            signature=signature,
            input_example=X_train.iloc[:3],
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        artifact = {
            "pipeline": pipeline,
            "feature_names": list(X.columns),
            "target_names": list(iris.target_names),
        }

        joblib.dump(artifact, MODEL_DIR / "iris_pipeline.joblib")

        metrics = {
            "accuracy": accuracy,
            "minimum_required_accuracy": MIN_ACCURACY,
            "passed_threshold": accuracy >= MIN_ACCURACY,
            "run_id": run.info.run_id,
            "model_uri": model_info.model_uri,
            "registered_model_name": REGISTERED_MODEL_NAME,
            "classification_report": classification_report(
                y_test,
                predictions,
                target_names=iris.target_names,
                output_dict=True,
            ),
        }

        (OUTPUT_DIR / "metrics.json").write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )

        print(f"MLflow run ID: {run.info.run_id}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Minimum required: {MIN_ACCURACY:.2f}")
        print(f"Registered model: {REGISTERED_MODEL_NAME}")
        print(f"Model URI: {model_info.model_uri}")
        print(f"Metrics saved to: {OUTPUT_DIR / 'metrics.json'}")

        if accuracy < MIN_ACCURACY:
            raise SystemExit(
                f"FAILED: accuracy {accuracy:.4f} "
                f"is below minimum {MIN_ACCURACY:.2f}"
            )

        print("PASS: model meets the minimum accuracy threshold.")


if __name__ == "__main__":
    main()
