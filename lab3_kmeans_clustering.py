"""
Lab 3: K-Means Clustering (Unsupervised Learning)
Dataset: sklearn.datasets.load_iris (features only)
Note: MLflow doesn't have built-in unsupervised evaluators,
      so we log metrics manually using silhouette score and inertia.
"""
import os
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from mlflow.models import infer_signature

# Silence Git warning in Docker environment
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

# Use environment variable or default to localhost for local development
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_registry_uri(tracking_uri)

experiment_name = "lab3-kmeans-clustering"
mlflow.set_experiment(experiment_name)

# Load Iris dataset (features only, no labels for clustering)
iris = load_iris(as_frame=True)
X = iris.data

# For unsupervised learning, we use all data (no train/test split needed)
# But we can still infer signature on a sample
X_sample = X.iloc[:50]  # Sample for signature inference

# Train K-Means model
n_clusters = 3
model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
model.fit(X)

with mlflow.start_run(run_name="kmeans_iris_clustering"):
    # Predict clusters for all data
    y_pred = model.predict(X)
    
    # Infer signature using sample (input features, output cluster labels)
    signature = infer_signature(X_sample, model.predict(X_sample))

    # Log model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
    )

    # Log parameters
    mlflow.log_params({
        "n_clusters": n_clusters,
        "random_state": 42,
        "n_init": 10,
    })

    # Calculate clustering metrics
    silhouette = silhouette_score(X, y_pred)
    davies_bouldin = davies_bouldin_score(X, y_pred)
    calinski_harabasz = calinski_harabasz_score(X, y_pred)
    inertia = model.inertia_

    # Log metrics
    mlflow.log_metrics({
        "silhouette_score": silhouette,
        "davies_bouldin_score": davies_bouldin,
        "calinski_harabasz_score": calinski_harabasz,
        "inertia": inertia,
    })

    # Print metrics
    print(f"Silhouette Score: {silhouette:.3f}")
    print(f"Davies-Bouldin Score: {davies_bouldin:.3f}")
    print(f"Calinski-Harabasz Score: {calinski_harabasz:.3f}")
    print(f"Inertia: {inertia:.3f}")
    print(f"Number of clusters: {n_clusters}")
    
    # Show cluster distribution
    unique, counts = np.unique(y_pred, return_counts=True)
    print(f"\nCluster distribution:")
    for cluster, count in zip(unique, counts):
        print(f"  Cluster {cluster}: {count} samples")

    print(f"\nAll logged metrics: silhouette_score, davies_bouldin_score, calinski_harabasz_score, inertia")

