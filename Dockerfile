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

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY classification.py .
COPY regressor.py .

# Set environment variables for MLflow
ENV MLFLOW_TRACKING_URI=http://mlflow-server:5000
ENV MLFLOW_REGISTRY_URI=http://mlflow-server:5000

# Default command (can be overridden in docker-compose)
CMD ["python", "classification.py"]

