import os
import mlflow
import mlflow.sklearn
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from mlflow.models import infer_signature
from mlflow.models import evaluate as mlflow_evaluate

# Use environment variable or default to localhost for local development
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_registry_uri(tracking_uri)
experiment_name = "evaluation"
mlflow.set_experiment(experiment_name)

X, y = shap.datasets.adult()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = xgb.XGBClassifier()
model.fit(X_train, y_train)

eval_data = X_test.copy()
eval_data["label"] = y_test

with mlflow.start_run(run_name="xgboost_adult_classifier"):
    y_pred = model.predict(X_test)
    signature = infer_signature(X_test, y_pred)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",  # Using artifact_path for compatibility
        signature=signature,
    )

    def predict_fn(df):
        features = df.drop(columns=["label"], errors="ignore")
        return model.predict(features)

    result = mlflow_evaluate(
        model=predict_fn,
        data=eval_data,
        targets="label",
        model_type="classifier",
        evaluators=["default"],
    )

    # Print metrics safely (handle missing metrics)
    if 'accuracy_score' in result.metrics:
        print(f"Accuracy: {result.metrics['accuracy_score']:.3f}")
    else:
        print("Accuracy: N/A")
    
    if 'f1_score' in result.metrics:
        print(f"F1 Score: {result.metrics['f1_score']:.3f}")
    else:
        print("F1 Score: N/A")
    
    # ROC AUC might not always be available (depends on evaluation configuration)
    if 'roc_auc_score' in result.metrics:
        print(f"ROC AUC: {result.metrics['roc_auc_score']:.3f}")
    elif 'roc_auc' in result.metrics:
        print(f"ROC AUC: {result.metrics['roc_auc']:.3f}")
    else:
        print("ROC AUC: Not available (check MLflow UI for all metrics)")
    
    # Print all available metrics for reference
    print(f"\nAll available metrics: {list(result.metrics.keys())}")
