# Deployment Steps - Complete Guide

## 🚀 Initial Deployment (First Time)

### Step 1: Stop Any Running Containers
```bash
docker-compose down
```

### Step 2: Build and Start All Services
```bash
docker-compose up -d --build
```

This will:
- Build all Docker images (including new API server)
- Start PostgreSQL
- Start MLflow Server
- Start MLflow App container
- Start MLflow API server (NEW)

### Step 3: Verify All Services are Running
```bash
docker-compose ps
```

You should see:
- `mlflow-postgres` - healthy
- `mlflow-server` - healthy
- `mlflow-app` - running
- `mlflow-api` - healthy (NEW)

### Step 4: Check Service Logs
```bash
# Check API server logs
docker-compose logs mlflow-api

# Check MLflow server logs
docker-compose logs mlflow-server

# Check all logs
docker-compose logs -f
```

### Step 5: Train All Models
**IMPORTANT**: Models must be trained before API can make predictions!

```bash
# Train all models
docker-compose exec mlflow-app bash run_all_experiments.sh
```

Or train individually:
```bash
docker-compose exec mlflow-app python lab1_logistic_regression.py
docker-compose exec mlflow-app python lab2_decision_tree_regressor.py
docker-compose exec mlflow-app python lab3_kmeans_clustering.py
docker-compose exec mlflow-app python lab4_random_forest_classifier.py
docker-compose exec mlflow-app python lab5_isolation_forest.py
docker-compose exec mlflow-app python classification.py
docker-compose exec mlflow-app python regressor.py
```

### Step 6: Verify API is Working
```bash
# Health check
curl http://localhost:8000/health

# Should return:
# {
#   "status": "healthy",
#   "models_loaded": 7,
#   "models": [...]
# }
```

### Step 7: Test a Prediction
```bash
# Test Lab 1 endpoint
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

---

## 🔄 Updating Deployment (After Code Changes)

### Scenario 1: Changed Python Code (training scripts or API)

```bash
# Stop services
docker-compose down

# Rebuild and restart
docker-compose up -d --build

# Retrain models if training scripts changed
docker-compose exec mlflow-app bash run_all_experiments.sh
```

### Scenario 2: Changed Dockerfile or docker-compose.yml

```bash
# Stop and remove containers
docker-compose down

# Rebuild images (no cache to ensure fresh build)
docker-compose build --no-cache

# Start services
docker-compose up -d
```

### Scenario 3: Only API Server Changed

```bash
# Rebuild only API service
docker-compose build mlflow-api

# Restart API service
docker-compose up -d mlflow-api

# Models will reload automatically on restart
```

### Scenario 4: Only Training Scripts Changed

```bash
# No need to restart - scripts are mounted as volume
# Just retrain models
docker-compose exec mlflow-app bash run_all_experiments.sh

# If you want to reload API with new models:
docker-compose restart mlflow-api
```

---

## 📋 Complete Deployment Checklist

### Initial Setup
- [ ] `docker-compose down` (if containers running)
- [ ] `docker-compose up -d --build`
- [ ] `docker-compose ps` (verify all services healthy)
- [ ] Train all models
- [ ] Test `/health` endpoint
- [ ] Test at least one prediction endpoint
- [ ] Access MLflow UI: `http://localhost:5000`
- [ ] Access API docs: `http://localhost:8000/docs`

### After Code Changes
- [ ] Determine what changed (API, training scripts, Docker config)
- [ ] Follow appropriate update scenario above
- [ ] Verify services are running
- [ ] Test endpoints

---

## 🔍 Verification Commands

### Check All Services Status
```bash
docker-compose ps
```

### Check Service Health
```bash
# PostgreSQL
docker-compose exec postgres pg_isready -U mlflow

# MLflow Server
curl http://localhost:5000/health

# API Server
curl http://localhost:8000/health
```

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f mlflow-api
docker-compose logs -f mlflow-server
docker-compose logs -f mlflow-app
```

### Check Models Loaded
```bash
curl http://localhost:8000/health | jq '.models'
```

---

## 🛑 Stopping Services

### Stop All Services (Keep Data)
```bash
docker-compose stop
```

### Stop and Remove Containers (Keep Volumes)
```bash
docker-compose down
```

### Stop and Remove Everything (⚠️ Deletes All Data)
```bash
docker-compose down -v
```

---

## 🔧 Troubleshooting Deployment

### Issue: API Server Not Starting

```bash
# Check logs
docker-compose logs mlflow-api

# Common causes:
# 1. MLflow server not ready - wait for it
# 2. Models not trained - train models first
# 3. Port 8000 already in use - change port in docker-compose.yml
```

### Issue: Models Not Loading

```bash
# Check if models are trained
# Open MLflow UI: http://localhost:5000
# Verify experiments exist

# Check API logs for loading errors
docker-compose logs mlflow-api | grep -i "error\|failed\|loading"
```

### Issue: Port Conflicts

If port 8000 is already in use, modify `docker-compose.yml`:
```yaml
ports:
  - "8001:8000"  # Change host port to 8001
```

---

## 📊 Service URLs

After deployment, access:

- **MLflow UI**: `http://localhost:5000`
- **API Server**: `http://localhost:8000`
- **API Docs**: `http://localhost:8000/docs`
- **API Health**: `http://localhost:8000/health`
- **PostgreSQL**: `localhost:5432` (from host)

---

## 🎯 Quick Reference

### Full Deployment (First Time)
```bash
docker-compose down
docker-compose up -d --build
docker-compose exec mlflow-app bash run_all_experiments.sh
curl http://localhost:8000/health
```

### Update After Code Changes
```bash
docker-compose down
docker-compose up -d --build
docker-compose exec mlflow-app bash run_all_experiments.sh
docker-compose restart mlflow-api
```

### Quick Restart (No Code Changes)
```bash
docker-compose restart
```

