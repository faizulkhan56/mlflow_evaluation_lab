# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Upgrade pip and install Python dependencies with increased timeout and retries
# Install in batches to avoid timeout on large packages
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --default-timeout=100 --retries 5 \
    mlflow>=2.0.0 scikit-learn>=1.0.0 && \
    pip install --no-cache-dir --default-timeout=100 --retries 5 \
    xgboost>=2.0.0 shap>=0.40.0 && \
    pip install --no-cache-dir --default-timeout=100 --retries 5 \
    pandas>=1.5.0 numpy>=1.24.0 psycopg2-binary>=2.9.0 && \
    pip install --no-cache-dir --default-timeout=100 --retries 5 \
    fastapi>=0.104.0 uvicorn[standard]>=0.24.0 pydantic>=2.0.0 requests>=2.31.0 && \
    pip install --no-cache-dir --default-timeout=100 --retries 5 \
    matplotlib>=3.5.0 seaborn>=0.12.0 kaggle>=1.5.0

# Copy application files
COPY classification.py .
COPY regressor.py .

# Set environment variables for MLflow
ENV MLFLOW_TRACKING_URI=http://mlflow-server:5000
ENV MLFLOW_REGISTRY_URI=http://mlflow-server:5000

# Default command (can be overridden in docker-compose)
CMD ["python", "classification.py"]

