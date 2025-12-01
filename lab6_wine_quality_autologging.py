"""
Lab 6: Wine Quality Classification with Autologging
Dataset: Wine Quality Red (UCI ML Repository)
Demonstrates MLflow autologging capabilities
"""
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mlflow.models import infer_signature
from mlflow.models import evaluate as mlflow_evaluate
import pandas as pd
import numpy as np

# Silence Git warning in Docker environment
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

# Use environment variable or default to localhost for local development
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_registry_uri(tracking_uri)

experiment_name = "lab6-wine-quality-autologging"
mlflow.set_experiment(experiment_name)

# Load Wine Quality dataset
# Uses Kaggle API if credentials available, otherwise falls back to UCI ML Repository
from dataset_loader import load_wine_quality_dataset
data = load_wine_quality_dataset()

# Prepare features and target
X = data.drop("quality", axis=1)
y = (data["quality"] >= 7).astype(int)  # Binary classification: quality >= 7 is "good"

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Enable autologging for sklearn
mlflow.sklearn.autolog()

# Train Random Forest Classifier with autologging
with mlflow.start_run(run_name="random_forest_autolog"):
    # Model parameters
    rf_params = {
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42,
    }
    
    # Train model (autologging will automatically log params, metrics, and model)
    model = RandomForestClassifier(**rf_params)
    model.fit(X_train, y_train)
    
    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate accuracy (autologging will log this automatically)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest (Autolog) Accuracy: {accuracy:.3f}")

    # Note: With autologging enabled, MLflow automatically logs:
    # - Model parameters (n_estimators, max_depth, etc.)
    # - Training metrics (if available)
    # - Model artifacts
    # - Model signature (input/output schema)
    # - Training time and other metadata
    
    # Infer signature for explicit logging (autologging also does this)
    signature = infer_signature(X_test, y_pred)
    
    # Log model with signature (autologging already logged it, but we ensure signature is included)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
    )
    
    # Prepare evaluation data for MLflow evaluation
    eval_data = X_test.copy()
    eval_data["label"] = y_test
    
    # Define prediction function for evaluation
    def predict_fn(df):
        features = df.drop(columns=["label"], errors="ignore")
        return model.predict(features)
    
    # Evaluate model using MLflow evaluators
    result = mlflow_evaluate(
        model=predict_fn,
        data=eval_data,
        targets="label",
        model_type="classifier",
        evaluators=["default"],
    )
    
    # Print metrics safely
    if "accuracy_score" in result.metrics:
        print(f"MLflow Evaluation Accuracy: {result.metrics['accuracy_score']:.3f}")
    else:
        print("Accuracy: N/A")
    
    if "f1_score" in result.metrics:
        print(f"F1 Score: {result.metrics['f1_score']:.3f}")
    else:
        print("F1 Score: N/A")
    
    if "roc_auc_score" in result.metrics:
        print(f"ROC AUC: {result.metrics['roc_auc_score']:.3f}")
    elif "roc_auc" in result.metrics:
        print(f"ROC AUC: {result.metrics['roc_auc']:.3f}")
    else:
        print("ROC AUC: Not available")
    
    if "precision_score" in result.metrics:
        print(f"Precision: {result.metrics['precision_score']:.3f}")
    
    if "recall_score" in result.metrics:
        print(f"Recall: {result.metrics['recall_score']:.3f}")
    
    # Print all available metrics
    print(f"\nAll available metrics: {list(result.metrics.keys())}")

# Disable autologging after the run
mlflow.sklearn.autolog(disable=True)

print("\n✓ Lab 6 (Autologging) completed successfully!")
print("Note: Check MLflow UI to see automatically logged parameters, metrics, and model artifacts.")

