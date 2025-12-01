"""
Lab 7: Wine Quality Classification with Manual Logging
Dataset: Wine Quality Red (UCI ML Repository)
Demonstrates manual MLflow logging (params, metrics, artifacts, model)
"""
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlflow.models import infer_signature
from mlflow.models import evaluate as mlflow_evaluate
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Silence Git warning in Docker environment
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

# Use environment variable or default to localhost for local development
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_registry_uri(tracking_uri)

experiment_name = "lab7-wine-quality-manual"
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

# Train Random Forest Classifier with manual logging
# Model parameters
rf_params = {
    "n_estimators": 100,
    "max_depth": 5,
    "random_state": 42,
    "criterion": "gini",
}

# Train model (outside run to make it accessible for evaluation run)
model = RandomForestClassifier(**rf_params)
model.fit(X_train, y_train)

# Generate predictions (outside run for reuse in evaluation)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

with mlflow.start_run(run_name="random_forest_manual"):
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Extract metrics from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Manual logging: Parameters
    mlflow.log_params(rf_params)
    
    # Manual logging: Metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("true_positives", float(tp))
    mlflow.log_metric("true_negatives", float(tn))
    mlflow.log_metric("false_positives", float(fp))
    mlflow.log_metric("false_negatives", float(fn))
    
    # Manual logging: Confusion matrix as artifact
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
    plt.title("Random Forest Manual Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    confusion_matrix_path = "rf_manual_confusion_matrix.png"
    plt.savefig(confusion_matrix_path, dpi=150, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(confusion_matrix_path)
    
    # Manual logging: Classification report as artifact
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_path = "rf_manual_classification_report.csv"
    report_df.to_csv(report_path)
    mlflow.log_artifact(report_path)
    
    # Manual logging: Feature importance plot
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x="importance", y="feature")
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    feature_importance_path = "rf_manual_feature_importance.png"
    plt.savefig(feature_importance_path, dpi=150, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(feature_importance_path)
    
    # Infer signature
    signature = infer_signature(X_test, y_pred)
    
    # Manual logging: Model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
    )
    
    # Log dataset info
    mlflow.log_param("dataset_size", len(data))
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))
    mlflow.log_param("n_features", len(X.columns))
    mlflow.log_param("target_distribution", f"Good: {y.sum()}, Bad: {len(y) - y.sum()}")
    
    print(f"Random Forest (Manual) Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")

# Prepare evaluation data for MLflow evaluation
eval_data = X_test.copy()
eval_data["label"] = y_test

# Start a new run to demonstrate MLflow evaluation with manually logged model
with mlflow.start_run(run_name="random_forest_manual_evaluation"):
    # Infer signature
    signature = infer_signature(X_test, y_pred)
    
    # Log model with signature
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
    )
    
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

# Clean up temporary files
for temp_file in [confusion_matrix_path, report_path, feature_importance_path]:
    if os.path.exists(temp_file):
        os.remove(temp_file)

print("\n✓ Lab 7 (Manual Logging) completed successfully!")

