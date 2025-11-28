"""
Lab 5: Isolation Forest (Anomaly Detection)
Dataset: sklearn.datasets.load_breast_cancer (malignant as anomalies)
Note: MLflow doesn't have built-in anomaly detection evaluators,
      so we log metrics manually using classification metrics on anomaly labels.
"""
import os
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from mlflow.models import infer_signature

# Silence Git warning in Docker environment
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

# Use environment variable or default to localhost for local development
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_registry_uri(tracking_uri)

experiment_name = "lab5-isolation-forest"
mlflow.set_experiment(experiment_name)

# Load breast cancer dataset
cancer = load_breast_cancer(as_frame=True)
X = cancer.data
y = cancer.target  # 0 = malignant (anomaly), 1 = benign (normal)

# Convert to anomaly labels: 0 = normal, 1 = anomaly
# Malignant (0) is treated as anomaly (1), Benign (1) is treated as normal (0)
y_anomaly = 1 - y  # Invert: malignant becomes 1 (anomaly), benign becomes 0 (normal)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_anomaly, test_size=0.2, random_state=42
)

# Train Isolation Forest
# contamination: proportion of outliers in the data
contamination = y_anomaly.mean()  # Use actual proportion of anomalies
model = IsolationForest(
    contamination=contamination,
    random_state=42,
    n_estimators=100
)
model.fit(X_train)

with mlflow.start_run(run_name="isolation_forest_breast_cancer"):
    # Predict anomalies: -1 = anomaly, 1 = normal
    y_pred_iso = model.predict(X_test)
    
    # Convert to binary: -1 -> 1 (anomaly), 1 -> 0 (normal)
    y_pred = (y_pred_iso == -1).astype(int)
    
    # Infer signature using X_test
    signature = infer_signature(X_test, y_pred)

    # Log model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
    )

    # Log parameters
    mlflow.log_params({
        "contamination": contamination,
        "random_state": 42,
        "n_estimators": 100,
    })

    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Log metrics
    mlflow.log_metrics({
        "accuracy_score": accuracy,
        "precision_score": precision,
        "recall_score": recall,
        "f1_score": f1,
    })

    # Print metrics
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"Contamination (anomaly rate): {contamination:.3f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix (rows=true, cols=predicted):")
    print("          Normal  Anomaly")
    print(f"Normal    {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"Anomaly   {cm[1,0]:6d}  {cm[1,1]:6d}")
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        print(f"\nTN (True Normal): {tn}, FP (False Positive/Anomaly): {fp}")
        print(f"FN (False Negative): {fn}, TP (True Anomaly): {tp}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))

    print(f"\nAll logged metrics: accuracy_score, precision_score, recall_score, f1_score")

