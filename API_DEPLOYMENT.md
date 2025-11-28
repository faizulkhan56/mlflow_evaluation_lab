# FastAPI Prediction Server - Deployment Guide

## Overview

The FastAPI server provides REST API endpoints for all 7 trained MLflow models, allowing you to make predictions via HTTP requests.

## Architecture

```
+------------------+
|   FastAPI Server |
|   (Port 8000)    |
+--------+---------+
         |
         v
+------------------+
|  MLflow Server   |
|  (Port 5000)     |
+--------+---------+
         |
         v
+------------------+
|   PostgreSQL     |
|  (Port 5432)     |
+------------------+
```

## Quick Start

### 1. Start All Services

```bash
docker-compose up -d
```

This starts:
- PostgreSQL database
- MLflow tracking server
- MLflow app container (for training)
- **FastAPI prediction server** (NEW)

### 2. Train Models First

Before using the API, you must train all models:

```bash
# Train all models
docker-compose exec mlflow-app bash run_all_experiments.sh

# Or train individually
docker-compose exec mlflow-app python lab1_logistic_regression.py
docker-compose exec mlflow-app python lab2_decision_tree_regressor.py
docker-compose exec mlflow-app python lab3_kmeans_clustering.py
docker-compose exec mlflow-app python lab4_random_forest_classifier.py
docker-compose exec mlflow-app python lab5_isolation_forest.py
docker-compose exec mlflow-app python classification.py
docker-compose exec mlflow-app python regressor.py
```

### 3. Verify API is Running

```bash
# Check health
curl http://localhost:8000/health

# Or open in browser
# http://localhost:8000/health
```

### 4. Access API Documentation

Open Swagger UI: `http://localhost:8000/docs`

This provides:
- Interactive API testing
- Request/response schemas
- Try-it-out functionality

## API Endpoints

| Endpoint | Method | Model | Type |
|----------|--------|-------|------|
| `/health` | GET | - | Health check |
| `/predict/lab1-logistic-regression` | POST | Logistic Regression | Classification |
| `/predict/lab2-decision-tree-regressor` | POST | Decision Tree | Regression |
| `/predict/lab3-kmeans-clustering` | POST | K-Means | Clustering |
| `/predict/lab4-random-forest` | POST | Random Forest | Classification |
| `/predict/lab5-isolation-forest` | POST | Isolation Forest | Anomaly Detection |
| `/predict/xgboost-classifier` | POST | XGBoost | Classification |
| `/predict/linear-regressor` | POST | Linear Regression | Regression |

## Testing with Postman

See `POSTMAN_EXAMPLES.md` for detailed examples with:
- Request body formats
- Response examples
- cURL commands
- Testing tips

## Testing with cURL

### Health Check
```bash
curl http://localhost:8000/health
```

### Example Prediction (Lab 1)
```bash
curl -X POST "http://localhost:8000/predict/lab1-logistic-regression" \
  -H "Content-Type: application/json" \
  -d '{
    "mean_radius": 17.99,
    "mean_texture": 10.38,
    "mean_perimeter": 122.8,
    "mean_area": 1001.0,
    "mean_smoothness": 0.1184,
    "mean_compactness": 0.2776,
    "mean_concavity": 0.3001,
    "mean_concave_points": 0.1471,
    "mean_symmetry": 0.2419,
    "mean_fractal_dimension": 0.07871,
    "radius_error": 1.095,
    "texture_error": 0.9053,
    "perimeter_error": 8.589,
    "area_error": 153.4,
    "smoothness_error": 0.006399,
    "compactness_error": 0.04904,
    "concavity_error": 0.05373,
    "concave_points_error": 0.01587,
    "symmetry_error": 0.03003,
    "fractal_dimension_error": 0.006193,
    "worst_radius": 25.38,
    "worst_texture": 17.33,
    "worst_perimeter": 184.6,
    "worst_area": 2019.0,
    "worst_smoothness": 0.1622,
    "worst_compactness": 0.6656,
    "worst_concavity": 0.7119,
    "worst_concave_points": 0.2654,
    "worst_symmetry": 0.4601,
    "worst_fractal_dimension": 0.1189
  }'
```

## How It Works

### Model Loading

1. **On Startup**: API server loads all models from MLflow
2. **Model Source**: Latest run from each experiment
3. **Caching**: Models cached in memory for fast predictions
4. **Error Handling**: If model not found, endpoint returns 503

### Prediction Flow

```
Client Request → FastAPI Endpoint → Model Cache → Prediction → Response
```

### Model Updates

After training a new model:
1. Stop API server: `docker-compose restart mlflow-api`
2. Models reload automatically on startup
3. New predictions use latest model version

## Troubleshooting

### Models Not Loading

**Error**: `503 Service Unavailable - Model not loaded`

**Solution**:
1. Verify models are trained: Check MLflow UI at `http://localhost:5000`
2. Check experiment names match exactly
3. Verify MLflow server is accessible from API container
4. Check API logs: `docker-compose logs mlflow-api`

### Connection Errors

**Error**: Cannot connect to MLflow server

**Solution**:
```bash
# Verify MLflow server is running
docker-compose ps mlflow-server

# Check network connectivity
docker-compose exec mlflow-api curl http://mlflow-server:5000/health

# Check logs
docker-compose logs mlflow-api
```

### Invalid Input Errors

**Error**: `422 Validation Error`

**Solution**:
- Check request body matches expected schema
- Verify all required fields are present
- Check data types (float vs int)
- See API docs at `/docs` for correct format

## Service Management

### View Logs
```bash
docker-compose logs -f mlflow-api
```

### Restart API
```bash
docker-compose restart mlflow-api
```

### Rebuild API (after code changes)
```bash
docker-compose build mlflow-api
docker-compose up -d mlflow-api
```

### Stop API
```bash
docker-compose stop mlflow-api
```

## Integration with Existing Services

The FastAPI server:
- ✅ **Does NOT break** existing training scripts
- ✅ **Shares** MLflow artifacts volume
- ✅ **Uses** same PostgreSQL database
- ✅ **Depends on** MLflow server (waits for it to be healthy)
- ✅ **Independent** from mlflow-app container

## Production Considerations

For production deployment:

1. **Authentication**: Add API keys or OAuth
2. **Rate Limiting**: Implement request throttling
3. **Model Versioning**: Use MLflow Model Registry
4. **Monitoring**: Add logging and metrics
5. **Scaling**: Use multiple API instances behind load balancer
6. **SSL/TLS**: Enable HTTPS
7. **Input Validation**: Enhanced validation for production data

## Next Steps

1. Train all models
2. Test API endpoints with Postman
3. Integrate API into your applications
4. Monitor predictions in MLflow UI
5. Set up CI/CD for model updates

