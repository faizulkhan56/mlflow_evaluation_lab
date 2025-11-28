#!/bin/bash
# Script to run all MLflow evaluation experiments
# Usage: docker-compose exec mlflow-app bash run_all_experiments.sh

echo "=========================================="
echo "Running All MLflow Evaluation Experiments"
echo "=========================================="
echo ""

echo "Lab 1: Logistic Regression (Classification)"
echo "-------------------------------------------"
python lab1_logistic_regression.py
echo ""

echo "Lab 2: Decision Tree Regressor (Regression)"
echo "-------------------------------------------"
python lab2_decision_tree_regressor.py
echo ""

echo "Lab 3: K-Means Clustering (Unsupervised)"
echo "-------------------------------------------"
python lab3_kmeans_clustering.py
echo ""

echo "Lab 4: Random Forest Classifier (Classification)"
echo "-------------------------------------------"
python lab4_random_forest_classifier.py
echo ""

echo "Lab 5: Isolation Forest (Anomaly Detection)"
echo "-------------------------------------------"
python lab5_isolation_forest.py
echo ""

echo "=========================================="
echo "All experiments completed!"
echo "View results at: http://localhost:5000"
echo "=========================================="

