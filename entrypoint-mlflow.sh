#!/bin/sh

echo "Starting MLflow server entrypoint..."

# Create mlruns directory if it doesn't exist
mkdir -p /app/mlruns
chmod 777 /app/mlruns 2>/dev/null || echo "Warning: Could not chmod /app/mlruns"

# Get PostgreSQL connection details from environment variables
POSTGRES_HOST="${POSTGRES_HOST:-postgres}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_USER="${POSTGRES_USER:-mlflow}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-mlflow}"
POSTGRES_DB="${POSTGRES_DB:-mlflow}"

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to be ready..."
until pg_isready -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" -U "${POSTGRES_USER}" > /dev/null 2>&1; do
  echo "PostgreSQL is unavailable - sleeping"
  sleep 1
done

echo "PostgreSQL is ready!"

# Construct PostgreSQL connection URI
POSTGRES_URI="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}"

echo "Using PostgreSQL database: ${POSTGRES_DB}"
echo "Database host: ${POSTGRES_HOST}:${POSTGRES_PORT}"
echo "Starting MLflow server..."

# Start MLflow server with PostgreSQL backend
exec mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri "${POSTGRES_URI}" \
  --artifacts-destination ./mlruns \
  --serve-artifacts \
  --allowed-hosts '*' \
  --cors-allowed-origins '*'

