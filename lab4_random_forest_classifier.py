"""
Lab 4: Random Forest Classifier (Classification)
Dataset: sklearn.datasets.load_iris
"""
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from mlflow.models import infer_signature
from mlflow.models import evaluate as mlflow_evaluate

# Silence Git warning in Docker environment
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

# Use environment variable or default to localhost for local development
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_registry_uri(tracking_uri)

experiment_name = "lab4-random-forest-classifier"
mlflow.set_experiment(experiment_name)

# Load Iris dataset
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Prepare evaluation data
eval_data = X_test.copy()
eval_data["label"] = y_test

with mlflow.start_run(run_name="random_forest_iris_classifier"):
    # Infer signature using X_test (represents real-world inference scenario)
    y_pred = model.predict(X_test)
    signature = infer_signature(X_test, y_pred)

    # Log model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
    )

    # Log parameters
    mlflow.log_params({
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42,
        "criterion": "gini",
    })

    # Define prediction function for evaluation
    def predict_fn(df):
        features = df.drop(columns=["label"], errors="ignore")
        return model.predict(features)

    # Evaluate model
    result = mlflow_evaluate(
        model=predict_fn,
        data=eval_data,
        targets="label",
        model_type="classifier",
        evaluators=["default"],
    )

    # Print metrics safely
    if "accuracy_score" in result.metrics:
        print(f"Accuracy: {result.metrics['accuracy_score']:.3f}")
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
        print("ROC AUC: Not available (multi-class may not support ROC AUC)")

    if "precision_score" in result.metrics:
        print(f"Precision: {result.metrics['precision_score']:.3f}")

    if "recall_score" in result.metrics:
        print(f"Recall: {result.metrics['recall_score']:.3f}")

    # Print all available metrics
    print(f"\nAll available metrics: {list(result.metrics.keys())}")

