from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris
from evidently import Report
from evidently.presets import DataDriftPreset

# Create the output directory
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the original Iris dataset as reference data
iris = load_iris(as_frame=True)
reference_data = iris.data.copy()

# Simulate drift by modifying two input features
current_data = reference_data.copy()
current_data["sepal length (cm)"] += 1.5
current_data["petal length (cm)"] += 1.0

# Save datasets used in the experiment
reference_data.to_csv(OUTPUT_DIR / "iris_reference_data.csv", index=False)
current_data.to_csv(OUTPUT_DIR / "iris_drifted_data.csv", index=False)

# Generate the Evidently data drift report
report = Report([
    DataDriftPreset()
])

result = report.run(
    reference_data=reference_data,
    current_data=current_data
)

# Save the HTML report
report_path = OUTPUT_DIR / "iris_drift_report.html"
result.save_html(str(report_path))

print("Data drift simulation completed successfully.")
print(f"Reference dataset: {len(reference_data)} rows")
print(f"Modified dataset: {len(current_data)} rows")
print(f"HTML report saved to: {report_path}")
