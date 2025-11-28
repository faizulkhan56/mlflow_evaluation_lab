# Quick Start Guide - All Experiments

## 🚀 Start Services

```bash
docker-compose up -d
```

## 🧪 Run All Experiments

```bash
# Option 1: Run all at once
docker-compose exec mlflow-app bash run_all_experiments.sh

# Option 2: Run individually
docker-compose exec mlflow-app python lab1_logistic_regression.py
docker-compose exec mlflow-app python lab2_decision_tree_regressor.py
docker-compose exec mlflow-app python lab3_kmeans_clustering.py
docker-compose exec mlflow-app python lab4_random_forest_classifier.py
docker-compose exec mlflow-app python lab5_isolation_forest.py
docker-compose exec mlflow-app python classification.py
docker-compose exec mlflow-app python regressor.py
```

## 📊 View Results

Open browser: `http://localhost:5000`

## 🛑 Stop Services

```bash
docker-compose down
```

## 📋 Experiment List

| File | Experiment Name | Type |
|------|----------------|------|
| `lab1_logistic_regression.py` | lab1-logistic-regression | Classification |
| `lab2_decision_tree_regressor.py` | lab2-decision-tree-regressor | Regression |
| `lab3_kmeans_clustering.py` | lab3-kmeans-clustering | Unsupervised |
| `lab4_random_forest_classifier.py` | lab4-random-forest-classifier | Classification |
| `lab5_isolation_forest.py` | lab5-isolation-forest | Anomaly Detection |
| `classification.py` | evaluation-classification | Classification |
| `regressor.py` | evaluation-regression | Regression |

