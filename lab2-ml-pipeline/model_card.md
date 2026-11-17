# Model Card — Iris Classification Pipeline

## Model details
- **Name:** Iris Logistic Regression Pipeline
- **Version:** 1.0
- **Task:** Multiclass classification of Iris flower species
- **Framework:** scikit-learn; exported to ONNX
- **Pipeline:** StandardScaler followed by LogisticRegression

## Intended use
Educational demonstration of a reproducible machine-learning workflow, model
serialization and interoperability. It is not intended for consequential
decisions.

## Data
The pipeline uses the built-in scikit-learn Iris dataset. It contains 150
records, four numeric measurements (sepal length, sepal width, petal length,
petal width) and three species classes.

## Evaluation
A stratified 80/20 train/test split is used with `random_state=42`. Run
`src/train_pipeline.py` to generate the actual accuracy and classification
metrics in `outputs/metrics.json`. Do not report a metric without running it.

## Limitations and risks
The dataset is small, clean and widely used for teaching. Performance on other
flower populations, measurement devices or distributions is not established.
Predictions depend on correct feature order and measurement units.

## Reproducibility
Dependencies are listed in `requirements.txt`. The serialized pipeline is
saved as `models/iris_pipeline.joblib`; ONNX export is saved as
`models/iris_pipeline.onnx`. `src/convert_onnx.py` compares class predictions
on ten records.
