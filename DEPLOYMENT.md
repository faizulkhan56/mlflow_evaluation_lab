# Deployment Guide - All MLflow Experiments

This guide explains how to run all 7 MLflow evaluation experiments in the project.

## 📋 Available Experiments

1. **Lab 1**: Logistic Regression (Classification) - `lab1_logistic_regression.py`
2. **Lab 2**: Decision Tree Regressor (Regression) - `lab2_decision_tree_regressor.py`
3. **Lab 3**: K-Means Clustering (Unsupervised) - `lab3_kmeans_clustering.py`
4. **Lab 4**: Random Forest Classifier (Classification) - `lab4_random_forest_classifier.py`
5. **Lab 5**: Isolation Forest (Anomaly Detection) - `lab5_isolation_forest.py`
6. **Original Classification**: XGBoost Classifier - `classification.py`
7. **Original Regression**: Linear Regression - `regressor.py`

## 🚀 Quick Start

### Prerequisites

Ensure Docker and Docker Compose are installed and running:

```bash
docker --version
docker-compose --version
```

### Step 1: Start All Services

```bash
docker-compose up -d
```

This starts:
- PostgreSQL database
- MLflow tracking server
- MLflow application container

### Step 2: Verify Services are Running

```bash
docker-compose ps
```

All services should show as "healthy" or "running".

### Step 3: Access MLflow UI

Open your browser: `http://localhost:5000`

## 🧪 Running Individual Experiments

### Lab 1: Logistic Regression

```bash
docker-compose exec mlflow-app python lab1_logistic_regression.py
```

**Expected Output:**
- Accuracy, F1 Score, ROC AUC, Precision, Recall
- Experiment: `lab1-logistic-regression`

### Lab 2: Decision Tree Regressor

```bash
docker-compose exec mlflow-app python lab2_decision_tree_regressor.py
```

**Expected Output:**
- MAE, RMSE, R², MSE
- Experiment: `lab2-decision-tree-regressor`

### Lab 3: K-Means Clustering

```bash
docker-compose exec mlflow-app python lab3_kmeans_clustering.py
```

**Expected Output:**
- Silhouette Score, Davies-Bouldin Score, Calinski-Harabasz Score, Inertia
- Experiment: `lab3-kmeans-clustering`

### Lab 4: Random Forest Classifier

```bash
docker-compose exec mlflow-app python lab4_random_forest_classifier.py
```

**Expected Output:**
- Accuracy, F1 Score, Precision, Recall
- Experiment: `lab4-random-forest-classifier`

### Lab 5: Isolation Forest (Anomaly Detection)

```bash
docker-compose exec mlflow-app python lab5_isolation_forest.py
```

**Expected Output:**
- Accuracy, Precision, Recall, F1 Score
- Confusion Matrix (Normal vs Anomaly)
- Experiment: `lab5-isolation-forest`

### Original Classification (XGBoost)

```bash
docker-compose exec mlflow-app python classification.py
```

**Expected Output:**
- Accuracy, F1 Score, ROC AUC
- Experiment: `evaluation-classification`

### Original Regression (Linear Regression)

```bash
docker-compose exec mlflow-app python regressor.py
```

**Expected Output:**
- MAE, RMSE, R²
- Experiment: `evaluation-regression`

## 🎯 Running All Experiments at Once

### Option 1: Using the Shell Script (Linux/macOS/Git Bash)

```bash
docker-compose exec mlflow-app bash run_all_experiments.sh
```

### Option 2: Using the Batch Script (Windows PowerShell)

```bash
docker-compose exec mlflow-app bash run_all_experiments.sh
```

(Note: The .bat file is for reference, but inside Docker container, use bash)

### Option 3: Manual Sequential Execution

```bash
# Run all experiments one by one
docker-compose exec mlflow-app python lab1_logistic_regression.py
docker-compose exec mlflow-app python lab2_decision_tree_regressor.py
docker-compose exec mlflow-app python lab3_kmeans_clustering.py
docker-compose exec mlflow-app python lab4_random_forest_classifier.py
docker-compose exec mlflow-app python lab5_isolation_forest.py
docker-compose exec mlflow-app python classification.py
docker-compose exec mlflow-app python regressor.py
```

## 📊 Viewing Results

### In MLflow UI

1. Open `http://localhost:5000` in your browser
2. You'll see all experiments listed:
   - `lab1-logistic-regression`
   - `lab2-decision-tree-regressor`
   - `lab3-kmeans-clustering`
   - `lab4-random-forest-classifier`
   - `lab5-isolation-forest`
   - `evaluation-classification`
   - `evaluation-regression`

3. Click on any experiment to see:
   - All runs
   - Metrics comparison
   - Model artifacts
   - Parameters logged

### Compare Experiments

In MLflow UI:
1. Go to "Compare" view
2. Select runs from different experiments
3. Compare metrics side-by-side

## 🔍 Troubleshooting

### Service Not Starting

```bash
# Check logs
docker-compose logs mlflow-server
docker-compose logs postgres

# Restart services
docker-compose restart
```

### Script Fails to Connect

```bash
# Verify MLflow server is healthy
docker-compose exec mlflow-app curl http://mlflow-server:5000/health

# Should return: {"status": "ok"}
```

### Database Connection Issues

```bash
# Check PostgreSQL is ready
docker-compose exec postgres pg_isready -U mlflow

# Check MLflow can connect
docker-compose logs mlflow-server | grep "PostgreSQL is ready"
```

## 🛑 Stopping Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (⚠️ deletes all data)
docker-compose down -v
```

## 📝 Notes

- **Data Persistence**: All experiments and models are stored in PostgreSQL and Docker volumes
- **Concurrent Runs**: You can run multiple experiments simultaneously
- **Resource Usage**: Each experiment uses CPU/memory; monitor with `docker stats`
- **Logs**: View real-time logs with `docker-compose logs -f mlflow-app`

## 🎓 Learning Path

Recommended order for learning:

1. **Start with Lab 1** (Logistic Regression) - Simple classification
2. **Then Lab 2** (Decision Tree) - Regression task
3. **Lab 4** (Random Forest) - More complex classification
4. **Lab 3** (K-Means) - Unsupervised learning
5. **Lab 5** (Isolation Forest) - Anomaly detection
6. **Original scripts** - Advanced examples with SHAP

Happy experimenting! 🚀

