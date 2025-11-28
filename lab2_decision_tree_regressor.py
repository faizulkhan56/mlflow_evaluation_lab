"""
Lab 2: Decision Tree Regressor (Regression)
Dataset: sklearn.datasets.fetch_california_housing
"""
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_california_housing
from mlflow.models import infer_signature
from mlflow.models import evaluate as mlflow_evaluate

# Silence Git warning in Docker environment
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

# Use environment variable or default to localhost for local development
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_registry_uri(tracking_uri)

experiment_name = "lab2-decision-tree-regressor"
mlflow.set_experiment(experiment_name)

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)
X = housing.data
y = housing.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Decision Tree Regressor
model = DecisionTreeRegressor(max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Prepare evaluation data
eval_data = X_test.copy()
eval_data["target"] = y_test

with mlflow.start_run(run_name="decision_tree_california_housing"):
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
        "max_depth": 10,
        "random_state": 42,
        "criterion": "squared_error",
    })

    # Define prediction function for evaluation
    def predict_fn(df):
        features = df.drop(columns=["target"], errors="ignore")
        return model.predict(features)

    # Evaluate model
    result = mlflow_evaluate(
        model=predict_fn,
        data=eval_data,
        targets="target",
        model_type="regressor",
        evaluators=["default"],
    )

    # Print metrics safely
    if "mean_absolute_error" in result.metrics:
        print(f"MAE:  {result.metrics['mean_absolute_error']:.3f}")
    else:
        print("MAE: N/A")

    if "root_mean_squared_error" in result.metrics:
        print(f"RMSE: {result.metrics['root_mean_squared_error']:.3f}")
    else:
        print("RMSE: N/A")

    if "r2_score" in result.metrics:
        print(f"R²:   {result.metrics['r2_score']:.3f}")
    else:
        print("R²: N/A")

    if "mean_squared_error" in result.metrics:
        print(f"MSE:  {result.metrics['mean_squared_error']:.3f}")

    # Print all available metrics
    print(f"\nAll available metrics: {list(result.metrics.keys())}")

