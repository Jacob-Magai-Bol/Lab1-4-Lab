# Lab1-4-Lab — End-to-end MLOps project

This repository implements Labs 1–4 using the Iris dataset and one consistent
four-feature classification problem.

## Requirements
- Python 3.10 or 3.11 (Python 3.11 is recommended for the Docker lab)
- Windows PowerShell, macOS Terminal, or Linux shell
- Docker Desktop for Lab 4
- A GitHub account and a GitHub Personal Access Token (PAT) for publishing to GHCR

## Project sequence
1. `lab1-model-training`: notebook and equivalent Python training script.
2. `lab2-ml-pipeline`: preprocessing/classification pipeline, Joblib, ONNX and
   ten-row prediction parity check.
3. `lab3-fastapi`: validated prediction API, pytest tests and Locust load test.
4. `lab4-containerisation`: baseline and slim Docker images, host-side API test,
   and GHCR publishing instructions.

Complete and verify each lab before moving to the next. Generated model files
are created by running the scripts; they are not committed in this starter project.

## Create an environment (Windows PowerShell)
From this directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Lab 1
```powershell
pip install -r lab1-model-training/requirements.txt
python lab1-model-training/src/train.py
```
Open `lab1-model-training/notebooks/iris_training.ipynb` in Jupyter or VS Code.
See `lab1-model-training/outputs/manual_steps.md` for the manual-step log.

## Lab 2
```powershell
pip install -r lab2-ml-pipeline/requirements.txt
python lab2-ml-pipeline/src/train_pipeline.py
python lab2-ml-pipeline/src/convert_onnx.py
```
The parity script prints the maximum absolute difference and whether the ten
predictions agree within tolerance.

## Lab 3
```powershell
pip install -r lab3-fastapi/requirements.txt
python lab2-ml-pipeline/src/train_pipeline.py
python lab2-ml-pipeline/src/convert_onnx.py
$env:PYTHONPATH="lab3-fastapi"
uvicorn app.main:app --reload
```
In a second terminal:
```powershell
python -m pytest lab3-fastapi/tests -v
```
Open `http://127.0.0.1:8000/docs` to try the API.

Run Locust:
```powershell
locust -f lab3-fastapi/locustfile.py --host http://127.0.0.1:8000
```
Open `http://localhost:8089`, set 20 users and a reasonable spawn rate (for
example, 5 users/second), then start the test. Record the p95 value shown in
Locust; latency depends on the computer and environment.

## Lab 4
Run commands from the repository root. First build the larger baseline image:
```powershell
docker build -f lab4-containerisation/Dockerfile -t iris-api:baseline .
docker image inspect iris-api:baseline --format='{{.Size}}'
```
Then build the slim image:
```powershell
docker build -f lab4-containerisation/Dockerfile.slim -t iris-api:slim .
docker image inspect iris-api:slim --format='{{.Size}}'
```
Run and test the slim image:
```powershell
docker run --rm -d --name iris-api -p 8000:8000 iris-api:slim
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict `
  -ContentType "application/json" `
  -Body '{"features":[5.1,3.5,1.4,0.2]}'
docker stop iris-api
```
Use `lab4-containerisation/README.md` for GHCR tagging and publishing.

## Important reporting notes
- Do not invent test outcomes, image sizes, or p95 latency. Run the commands
  and record the values from your own machine.
- ONNX and Joblib predictions are floating-point values; compare using a
  tolerance rather than requiring bit-for-bit equality.
- This is an educational Iris classifier, not a production medical or
  high-stakes decision system.
