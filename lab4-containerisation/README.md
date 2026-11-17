# Lab 4 — Containerisation of the FastAPI ML API

## 1. Objective

The objective of this lab was to containerise the FastAPI machine-learning API developed in Lab 3, measure the original Docker image size, optimise the image using a slim Python base image and `.dockerignore`, run the container, and test the prediction endpoint from the host machine.

## 2. Docker Images

Two Docker images were built for comparison.

### 2.1 Baseline Image

- **Base image:** `python:3.11`
- **Image tag:** `iris-api:baseline`
- **Build command:**

```powershell
docker build -f lab4-containerisation/Dockerfile -t iris-api:baseline .
```

### 2.2 Optimised Slim Image

- **Base image:** `python:3.11-slim`
- **Image tag:** `iris-api:slim`
- **Build command:**

```powershell
docker build -f lab4-containerisation/Dockerfile.slim -t iris-api:slim .
```

The optimised Dockerfile uses `--no-cache-dir` to prevent pip from retaining installation cache files.

## 3. Docker Image Size Comparison

The following measurements were obtained from Docker:

| Image               | Disk Usage | Content Size |
| ------------------- | ---------: | -----------: |
| `iris-api:baseline` |    2.04 GB |       510 MB |
| `iris-api:slim`     |     612 MB |       141 MB |

### 3.1 Image Size Reduction

The percentage reduction in content size was calculated as follows:

\[
\text{Reduction} = \frac{510-141}{510}\times100
\]

\[
\text{Reduction} = 72.35\%
\]

The optimised slim image reduced the content size by approximately **72.4%** compared with the baseline image.

## 4. Docker Ignore Configuration

The project root `.dockerignore` file excludes unnecessary files and directories from the Docker build context.

Excluded items include:

- Python virtual environments
- Python cache files
- Git repository files
- Test directories
- Jupyter notebooks
- Training outputs
- Metrics files
- ONNX files

This reduces the build context and prevents unnecessary project files from being included in the Docker build.

## 5. Running the Container

The slim image was started using the following command:

```powershell
docker run -d --name iris-api-container -p 8000:8000 -e MODEL_PATH=/app/models/iris_pipeline.joblib iris-api:slim
```

The container status was checked using:

```powershell
docker ps -a --filter "name=iris-api-container"
```

**Observed result:**

```text
IMAGE: iris-api:slim
STATUS: Up
PORTS: 0.0.0.0:8000->8000/tcp
```

The output confirmed that the container was running and port 8000 was mapped to the host machine.

## 6. Testing the API from the Host Machine

### 6.1 Health Endpoint

The health endpoint was tested using:

```powershell
Invoke-RestMethod -Uri http://localhost:8000/health
```

**Observed result:**

```text
status model_loaded
ok     True
```

This confirmed that the container was running and the machine-learning model was loaded successfully.

### 6.2 Prediction Endpoint

The prediction endpoint was tested using the following PowerShell commands:

```powershell
$body = @{
    features = @(5.1, 3.5, 1.4, 0.2)
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -ContentType "application/json" -Body $body
```

**Observed result:**

```text
predicted_class species
--------------- -------
              0 setosa
```

The API successfully received the prediction request from the host machine and returned the expected Iris classification.

## 7. Conclusion

The Lab 3 FastAPI machine-learning service was successfully containerised using Docker. Both the baseline and optimised images were built and compared.

The slim image reduced the content size by approximately 72.4%. The `.dockerignore` file also helped reduce the Docker build context by excluding unnecessary project files.

The container ran successfully, the health endpoint confirmed that the model was loaded, and the prediction endpoint returned the expected result when accessed from the host machine.

**Remaining task:** Publish the Docker image to GitHub Container Registry (GHCR), as required by the lab submission instructions.
