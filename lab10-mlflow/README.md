# Lab 10 — MLflow, Model Registry, CI/CD, and Docker

## Overview

This project implements an end-to-end MLOps pipeline for an Iris classification model.

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

## Model Performance

| Metric | Value |
|---|---|
| Model | IrisClassifier |
| Algorithm | Scikit-learn classification pipeline |
| Accuracy | 93.33% |
| Minimum required accuracy | 90% |
| Result | PASS |

## Project Structure

```text
lab10-mlflow/
├── src/
│   ├── api/
│   │   └── main.py
│   └── train.py
├── models/
│   └── iris_pipeline.joblib
├── outputs/
│   └── metrics.json
├── tests/
│   └── test_api.py
├── requirements.txt
├── requirements-runtime.txt
├── Dockerfile
├── .dockerignore
└── README.md
