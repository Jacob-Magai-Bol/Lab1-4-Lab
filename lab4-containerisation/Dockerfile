# Baseline image for comparison; intentionally uses the full Python image.
FROM python:3.11
WORKDIR /app
COPY lab4-containerisation/requirements-runtime.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY lab3-fastapi/app /app/app
COPY lab2-ml-pipeline/models/iris_pipeline.joblib /app/models/iris_pipeline.joblib
ENV MODEL_PATH=/app/models/iris_pipeline.joblib
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
