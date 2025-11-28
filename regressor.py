import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from mlflow.models import infer_signature
from mlflow.models import evaluate as mlflow_evaluate
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

# Use environment variable or default to localhost for local development
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_registry_uri(tracking_uri)

experiment_name = "evaluation2"
mlflow.set_experiment(experiment_name)

housing = fetch_california_housing(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, test_size=0.2, random_state=42
)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

eval_data = X_test.copy()
eval_data["target"] = y_test

with mlflow.start_run(run_name="linreg_california_housing"):
    y_pred = reg_model.predict(X_train)
    signature = infer_signature(X_train, y_pred)

    mlflow.sklearn.log_model(
        sk_model=reg_model,
        artifact_path="model",  # Using artifact_path for compatibility
        signature=signature,
    )

    def predict_fn(df):
        features = df.drop(columns=["target"], errors="ignore")
        return reg_model.predict(features)

    result = mlflow_evaluate(
        model=predict_fn,
        data=eval_data,
        targets="target",
        model_type="regressor",
        evaluators=["default"],
    )

    # Print metrics safely (handle missing metrics)
    if 'mean_absolute_error' in result.metrics:
        print(f"MAE:  {result.metrics['mean_absolute_error']:.3f}")
    else:
        print("MAE: N/A")
    
    if 'root_mean_squared_error' in result.metrics:
        print(f"RMSE: {result.metrics['root_mean_squared_error']:.3f}")
    else:
        print("RMSE: N/A")
    
    if 'r2_score' in result.metrics:
        print(f"R²:   {result.metrics['r2_score']:.3f}")
    else:
        print("R²: N/A")
    
    # Print all available metrics for reference
    print(f"\nAll available metrics: {list(result.metrics.keys())}")
