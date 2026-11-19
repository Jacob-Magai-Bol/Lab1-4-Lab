# Lab 10 and Lab 11 — MLOps, Monitoring, and Data Drift Detection

## 1. Overview

This project implements an end-to-end MLOps pipeline for an Iris classification model.

The project covers model development, experiment tracking, deployment, containerization, continuous integration, monitoring, and data drift detection.

### Lab 10 — MLOps Pipeline

The pipeline includes:
- Model training using Scikit-learn
- Experiment tracking using MLflow
- Model registration in the MLflow Model Registry
- Model performance evaluation against a minimum accuracy threshold
- FastAPI prediction service
- Automated API tests using pytest
- Docker containerization
- GitHub Actions CI/CD
- Container image publishing to GitHub Container Registry (GHCR)

### Lab 11 — Monitoring and Data Drift Detection

The monitoring implementation includes:
- Prometheus metrics integration using `prometheus-client`
- A `/metrics` endpoint in FastAPI
- HTTP request counters and latency measurements
- Iris prediction counters by species
- Prometheus metrics collection
- Grafana dashboard visualization
- Data drift simulation using modified Iris data
- Evidently data drift analysis
- HTML drift report generation

## 2. Model Performance

| Metric | Value |
|---|---|
| Model | IrisClassifier |
| Algorithm | Scikit-learn classification pipeline |
| Accuracy | 93.33% |
| Minimum required accuracy | 90% |
| Result | PASS |

## 3. Monitoring Metrics

The FastAPI service exposes the following Prometheus metrics:

| Metric | Description |
|---|---|
| `http_requests_total` | Total HTTP requests |
| `http_request_duration_seconds` | HTTP request latency |
| `iris_predictions_total` | Total successful predictions grouped by species |

### API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Checks API health |
| `/predict` | POST | Predicts Iris species |
| `/metrics` | GET | Exposes Prometheus metrics |

## 4. Prometheus and Grafana Setup

The monitoring services are managed using Docker Compose.

### Start the monitoring services

From the repository root, run:

```powershell
docker compose -f .\lab10-mlflow\docker-compose.yml up -d
```

### Access the services

| Service | URL |
|---|---|
| FastAPI | http://localhost:8000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

Configure Grafana with Prometheus as its data source:

```text
http://prometheus:9090
```

### Prometheus queries

Prediction count:

```promql
iris_predictions_total
```

Average API latency in milliseconds:

```promql
1000 * (
  http_request_duration_seconds_sum
  /
  http_request_duration_seconds_count
)
```

These queries allow the monitoring dashboard to display prediction activity and API response latency.

## 5. Data Drift Simulation Using Evidently

Evidently is used to compare the original Iris dataset against a modified dataset to identify changes in feature distributions.

### Simulation method

The original Iris dataset is used as the reference dataset.

A second dataset is created by modifying two features:

- Sepal length is increased by 1.5.
- Petal length is increased by 1.0.

The two datasets are then compared using Evidently's `DataDriftPreset`.

### Run the simulation

Install the dependencies:

```powershell
python -m pip install -r .\lab10-mlflow\requirements.txt
```

Run the drift simulation:

```powershell
python .\lab10-mlflow\drift_simulation.py
```

The generated HTML report is saved at:

```text
lab10-mlflow/outputs/iris_drift_report.html
```

## 6. Data Drift Results

The Evidently report detected dataset drift.

| Measurement | Result |
|---|---|
| Total features | 4 |
| Drifted features | 2 |
| Share of drifted features | 50% |
| Detection threshold | 0.5 |
| Overall dataset drift | Detected |

### Feature-level results

| Feature | Drift Status |
|---|---|
| Sepal length | Detected |
| Petal length | Detected |
| Sepal width | Not detected |
| Petal width | Not detected |

The results confirm that Evidently successfully identified the two deliberately modified features.

## 7. Recommendation

The Evidently data drift report detected significant changes in 50% of the Iris dataset features, with drift identified in sepal length and petal length, while sepal width and petal width remained stable. This indicates that the incoming data distribution has changed compared with the reference dataset. I recommend continuously monitoring incoming data using Evidently and Prometheus, investigating the causes of detected changes, and validating model performance using newly collected labelled data. If the drift negatively affects prediction accuracy, the model should be retrained using representative and updated data before deployment. Regular monitoring and evaluation will help maintain the reliability and performance of the Iris Classification API.

## 8. Automated Testing

The project uses pytest to verify the API functionality.

Run the tests:

```powershell
python -m pytest .\lab10-mlflow\tests -v
```

### Test results

| Test | Result |
|---|---|
| Health endpoint | PASSED |
| Valid prediction | PASSED |
| Invalid prediction | PASSED |
| Prometheus metrics endpoint | PASSED |

Total: 4 tests passed.

## 9. Project Structure

```text
lab10-mlflow/
├── src/
│   ├── api/
│   │   └── main.py
│   └── train.py
├── models/
│   └── iris_pipeline.joblib
├── monitoring/
│   └── prometheus.yml
├── outputs/
│   ├── metrics.json
│   ├── iris_reference_data.csv
│   ├── iris_drifted_data.csv
│   └── iris_drift_report.html
├── tests/
│   └── test_api.py
├── drift_simulation.py
├── docker-compose.yml
├── requirements.txt
├── requirements-runtime.txt
├── Dockerfile
├── .dockerignore
└── README.md

```

## 10. Technologies Used

- Python
- Scikit-learn
- MLflow
- FastAPI
- Prometheus
- Grafana
- Evidently AI
- Docker
- Docker Compose
- pytest
- GitHub Actions
- Git and GitHub

## 11. Conclusion

Lab 10 established the Iris classification MLOps pipeline, including model training, registration, API deployment, testing, and containerization.

Lab 11 extended the pipeline with Prometheus monitoring, Grafana visualization, and Evidently data drift detection. The successful detection of drift in two out of four features demonstrates how data monitoring can help identify changes that may affect model reliability.