"""
FastAPI Server for MLflow Model Predictions
Provides prediction endpoints for all 7 trained models
"""
import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Any, Union
import logging
import asyncio

# Silence Git warning
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MLflow Model Prediction API",
    description="Prediction endpoints for all trained MLflow models",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Global model cache
models_cache = {}
models_loading = {}  # Track which models are currently being loaded

def load_model_from_latest_run(experiment_name: str, run_name_pattern: str = None):
    """Load the latest model from an MLflow experiment"""
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        
        # Get all runs from the experiment
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time desc"], max_results=1)
        
        if runs.empty:
            raise ValueError(f"No runs found in experiment '{experiment_name}'")
        
        latest_run_id = runs.iloc[0]["run_id"]
        model_uri = f"runs:/{latest_run_id}/model"
        
        logger.info(f"Loading model from run {latest_run_id} in experiment {experiment_name}")
        model = mlflow.sklearn.load_model(model_uri)
        return model
    except Exception as e:
        logger.error(f"Error loading model from {experiment_name}: {str(e)}")
        raise

def ensure_model_loaded(model_key: str, experiment_name: str):
    """Ensure a model is loaded, loading it lazily if needed"""
    if model_key in models_cache:
        return True
    
    # Check if model is currently being loaded
    if models_loading.get(model_key, False):
        raise HTTPException(status_code=503, detail=f"Model '{model_key}' is currently loading. Please try again in a moment.")
    
    # Try to load the model
    try:
        logger.info(f"Lazy loading {model_key} from {experiment_name}...")
        models_loading[model_key] = True
        models_cache[model_key] = load_model_from_latest_run(experiment_name)
        models_loading[model_key] = False
        logger.info(f"✓ {model_key} loaded successfully")
        return True
    except Exception as e:
        models_loading[model_key] = False
        logger.error(f"Failed to load {model_key}: {str(e)}")
        raise HTTPException(
            status_code=503, 
            detail=f"Model '{model_key}' not available. Please ensure the experiment '{experiment_name}' exists and has at least one run with a logged model. Error: {str(e)}"
        )

def map_breast_cancer_features(feature_dict: dict) -> dict:
    """Map API feature names (underscores) to sklearn feature names (spaces)"""
    # sklearn breast cancer dataset uses spaces instead of underscores
    mapping = {
        "mean_radius": "mean radius",
        "mean_texture": "mean texture",
        "mean_perimeter": "mean perimeter",
        "mean_area": "mean area",
        "mean_smoothness": "mean smoothness",
        "mean_compactness": "mean compactness",
        "mean_concavity": "mean concavity",
        "mean_concave_points": "mean concave points",
        "mean_symmetry": "mean symmetry",
        "mean_fractal_dimension": "mean fractal dimension",
        "radius_error": "radius error",
        "texture_error": "texture error",
        "perimeter_error": "perimeter error",
        "area_error": "area error",
        "smoothness_error": "smoothness error",
        "compactness_error": "compactness error",
        "concavity_error": "concavity error",
        "concave_points_error": "concave points error",
        "symmetry_error": "symmetry error",
        "fractal_dimension_error": "fractal dimension error",
        "worst_radius": "worst radius",
        "worst_texture": "worst texture",
        "worst_perimeter": "worst perimeter",
        "worst_area": "worst area",
        "worst_smoothness": "worst smoothness",
        "worst_compactness": "worst compactness",
        "worst_concavity": "worst concavity",
        "worst_concave_points": "worst concave points",
        "worst_symmetry": "worst symmetry",
        "worst_fractal_dimension": "worst fractal dimension",
    }
    return {mapping[k]: v for k, v in feature_dict.items() if k in mapping}

def load_model_sync(model_key: str, experiment_name: str):
    """Synchronous model loading function"""
    try:
        logger.info(f"Loading {model_key} from {experiment_name}...")
        models_cache[model_key] = load_model_from_latest_run(experiment_name)
        logger.info(f"✓ {model_key} loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load {model_key} model: {e}")
    finally:
        if model_key in models_loading:
            models_loading[model_key] = False

def load_models_background_sync():
    """Load all models in background thread (non-blocking)"""
    logger.info("Starting background model loading from MLflow...")
    
    model_configs = [
        ("lab1_logistic_regression", "lab1-logistic-regression"),
        ("lab2_decision_tree_regressor", "lab2-decision-tree-regressor"),
        ("lab3_kmeans_clustering", "lab3-kmeans-clustering"),
        ("lab4_random_forest", "lab4-random-forest-classifier"),
        ("lab5_isolation_forest", "lab5-isolation-forest"),
        ("xgboost_classifier", "evaluation-classification"),
        ("linear_regressor", "evaluation-regression"),
    ]
    
    # Initialize loading flags
    for model_key, _ in model_configs:
        models_loading[model_key] = True
    
    # Load models sequentially in background thread
    for model_key, experiment_name in model_configs:
        load_model_sync(model_key, experiment_name)
    
    logger.info(f"Background model loading completed. {len(models_cache)} models loaded.")

# Load models on startup (non-blocking)
@app.on_event("startup")
async def startup_event():
    """Startup event - initiate background model loading"""
    # Start model loading in background thread (non-blocking)
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, load_models_background_sync)
    logger.info("Server started. Models loading in background...")

# Pydantic models for request validation

# Lab 1: Logistic Regression - Breast Cancer (30 features)
class BreastCancerFeatures(BaseModel):
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float
    mean_compactness: float
    mean_concavity: float
    mean_concave_points: float
    mean_symmetry: float
    mean_fractal_dimension: float
    radius_error: float
    texture_error: float
    perimeter_error: float
    area_error: float
    smoothness_error: float
    compactness_error: float
    concavity_error: float
    concave_points_error: float
    symmetry_error: float
    fractal_dimension_error: float
    worst_radius: float
    worst_texture: float
    worst_perimeter: float
    worst_area: float
    worst_smoothness: float
    worst_compactness: float
    worst_concavity: float
    worst_concave_points: float
    worst_symmetry: float
    worst_fractal_dimension: float

# Lab 2 & Regressor: California Housing (8 features)
class CaliforniaHousingFeatures(BaseModel):
    MedInc: float = Field(..., description="Median income in block group")
    HouseAge: float = Field(..., description="Median house age in block group")
    AveRooms: float = Field(..., description="Average number of rooms per household")
    AveBedrms: float = Field(..., description="Average number of bedrooms per household")
    Population: float = Field(..., description="Block group population")
    AveOccup: float = Field(..., description="Average number of household members")
    Latitude: float = Field(..., description="Block group latitude")
    Longitude: float = Field(..., description="Block group longitude")

# Lab 3 & Lab 4: Iris (4 features)
class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., description="Sepal length in cm")
    sepal_width: float = Field(..., description="Sepal width in cm")
    petal_length: float = Field(..., description="Petal length in cm")
    petal_width: float = Field(..., description="Petal width in cm")

# Lab 5: Isolation Forest - uses same as Lab 1 (Breast Cancer features)
# XGBoost: Adult dataset (14 features - simplified for API)
class AdultFeatures(BaseModel):
    age: float
    workclass: str
    fnlwgt: float
    education: str
    education_num: float
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: float
    capital_loss: float
    hours_per_week: float
    native_country: str

# Response models
class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra='allow')  # Allow extra fields like prediction_label, prediction_probability, etc.
    
    prediction: Any  # Can be int, float, list, etc. depending on model type
    model_name: str
    prediction_type: str

class HealthResponse(BaseModel):
    status: str
    models_loaded: int
    models: List[str]

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint - returns healthy even if models are still loading"""
    return {
        "status": "healthy",
        "models_loaded": len(models_cache),
        "models": list(models_cache.keys())
    }

# Lab 1: Logistic Regression - Breast Cancer Classification
@app.post("/predict/lab1-logistic-regression", response_model=PredictionResponse)
async def predict_lab1_logistic_regression(features: BreastCancerFeatures):
    """
    Predict breast cancer diagnosis using Logistic Regression
    Returns: 0 (malignant) or 1 (benign)
    """
    # Ensure model is loaded (lazy loading)
    ensure_model_loaded("lab1_logistic_regression", "lab1-logistic-regression")
    
    try:
        # Convert to DataFrame with correct column names (map underscores to spaces)
        feature_dict = features.dict()
        mapped_features = map_breast_cancer_features(feature_dict)
        df = pd.DataFrame([mapped_features])
        
        # Predict
        prediction = models_cache["lab1_logistic_regression"].predict(df)[0]
        prediction_proba = models_cache["lab1_logistic_regression"].predict_proba(df)[0]
        
        return {
            "prediction": int(prediction),
            "prediction_label": "benign" if prediction == 1 else "malignant",
            "prediction_probability": {
                "malignant": float(prediction_proba[0]),
                "benign": float(prediction_proba[1])
            },
            "model_name": "lab1-logistic-regression",
            "prediction_type": "classification"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Lab 2: Decision Tree Regressor - California Housing
@app.post("/predict/lab2-decision-tree-regressor", response_model=PredictionResponse)
async def predict_lab2_decision_tree(features: CaliforniaHousingFeatures):
    """
    Predict median house value using Decision Tree Regressor
    Returns: Predicted median house value
    """
    # Ensure model is loaded (lazy loading)
    ensure_model_loaded("lab2_decision_tree_regressor", "lab2-decision-tree-regressor")
    
    try:
        feature_dict = features.dict()
        df = pd.DataFrame([feature_dict])
        
        prediction = models_cache["lab2_decision_tree_regressor"].predict(df)[0]
        
        return {
            "prediction": float(prediction),
            "model_name": "lab2-decision-tree-regressor",
            "prediction_type": "regression",
            "unit": "median house value (in hundreds of thousands)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Lab 3: K-Means Clustering - Iris
@app.post("/predict/lab3-kmeans-clustering", response_model=PredictionResponse)
async def predict_lab3_kmeans(features: IrisFeatures):
    """
    Predict cluster assignment using K-Means
    Returns: Cluster label (0, 1, or 2)
    """
    # Ensure model is loaded (lazy loading)
    ensure_model_loaded("lab3_kmeans_clustering", "lab3-kmeans-clustering")
    
    try:
        feature_dict = features.dict()
        # Map API feature names to sklearn Iris dataset feature names
        # sklearn uses "sepal length (cm)", "sepal width (cm)", etc.
        df = pd.DataFrame([{
            "sepal length (cm)": feature_dict["sepal_length"],
            "sepal width (cm)": feature_dict["sepal_width"],
            "petal length (cm)": feature_dict["petal_length"],
            "petal width (cm)": feature_dict["petal_width"]
        }])
        
        prediction = models_cache["lab3_kmeans_clustering"].predict(df)[0]
        
        return {
            "prediction": int(prediction),
            "cluster_label": f"cluster_{int(prediction)}",
            "model_name": "lab3-kmeans-clustering",
            "prediction_type": "clustering"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Lab 4: Random Forest Classifier - Iris
@app.post("/predict/lab4-random-forest", response_model=PredictionResponse)
async def predict_lab4_random_forest(features: IrisFeatures):
    """
    Predict iris species using Random Forest
    Returns: 0 (setosa), 1 (versicolor), or 2 (virginica)
    """
    # Ensure model is loaded (lazy loading)
    ensure_model_loaded("lab4_random_forest", "lab4-random-forest-classifier")
    
    try:
        feature_dict = features.dict()
        # Map API feature names to sklearn Iris dataset feature names
        # sklearn uses "sepal length (cm)", "sepal width (cm)", etc.
        df = pd.DataFrame([{
            "sepal length (cm)": feature_dict["sepal_length"],
            "sepal width (cm)": feature_dict["sepal_width"],
            "petal length (cm)": feature_dict["petal_length"],
            "petal width (cm)": feature_dict["petal_width"]
        }])
        
        prediction = models_cache["lab4_random_forest"].predict(df)[0]
        prediction_proba = models_cache["lab4_random_forest"].predict_proba(df)[0]
        
        species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
        
        return {
            "prediction": int(prediction),
            "prediction_label": species_map[int(prediction)],
            "prediction_probability": {
                species_map[i]: float(prediction_proba[i]) for i in range(3)
            },
            "model_name": "lab4-random-forest",
            "prediction_type": "classification"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Lab 5: Isolation Forest - Anomaly Detection
@app.post("/predict/lab5-isolation-forest", response_model=PredictionResponse)
async def predict_lab5_isolation_forest(features: BreastCancerFeatures):
    """
    Detect anomalies using Isolation Forest
    Returns: -1 (anomaly) or 1 (normal)
    """
    # Ensure model is loaded (lazy loading)
    ensure_model_loaded("lab5_isolation_forest", "lab5-isolation-forest")
    
    try:
        # Convert to DataFrame with correct column names (map underscores to spaces)
        feature_dict = features.dict()
        mapped_features = map_breast_cancer_features(feature_dict)
        df = pd.DataFrame([mapped_features])
        
        prediction = models_cache["lab5_isolation_forest"].predict(df)[0]
        
        return {
            "prediction": int(prediction),
            "prediction_label": "anomaly" if prediction == -1 else "normal",
            "model_name": "lab5-isolation-forest",
            "prediction_type": "anomaly_detection"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# XGBoost Classifier - Adult Dataset
# Note: This model uses categorical features. For simplicity, accept as dict
@app.post("/predict/xgboost-classifier")
async def predict_xgboost_classifier(features: dict):
    """
    Predict income level using XGBoost Classifier
    Accepts: Dictionary with all features from adult dataset
    Returns: 0 (<=50K) or 1 (>50K)
    
    Required features (from shap.datasets.adult()):
    - Age, Workclass, Fnlwgt, Education, Education-Num, Marital-Status,
      Occupation, Relationship, Race, Sex, Capital-Gain, Capital-Loss,
      Hours-per-week, Native-Country
    """
    # Ensure model is loaded (lazy loading)
    ensure_model_loaded("xgboost_classifier", "evaluation-classification")
    
    try:
        # Convert dict to DataFrame (XGBoost handles categorical features)
        df = pd.DataFrame([features])
        
        prediction = models_cache["xgboost_classifier"].predict(df)[0]
        prediction_proba = models_cache["xgboost_classifier"].predict_proba(df)[0]
        
        return {
            "prediction": int(prediction),
            "prediction_label": ">50K" if prediction == 1 else "<=50K",
            "prediction_probability": {
                "<=50K": float(prediction_proba[0]),
                ">50K": float(prediction_proba[1])
            },
            "model_name": "xgboost-classifier",
            "prediction_type": "classification"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Linear Regressor - California Housing
@app.post("/predict/linear-regressor", response_model=PredictionResponse)
async def predict_linear_regressor(features: CaliforniaHousingFeatures):
    """
    Predict median house value using Linear Regression
    Returns: Predicted median house value
    """
    # Ensure model is loaded (lazy loading)
    ensure_model_loaded("linear_regressor", "evaluation-regression")
    
    try:
        feature_dict = features.dict()
        df = pd.DataFrame([feature_dict])
        
        prediction = models_cache["linear_regressor"].predict(df)[0]
        
        return {
            "prediction": float(prediction),
            "model_name": "linear-regressor",
            "prediction_type": "regression",
            "unit": "median house value (in hundreds of thousands)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    """API information"""
    return {
        "message": "MLflow Model Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "lab1": "/predict/lab1-logistic-regression",
            "lab2": "/predict/lab2-decision-tree-regressor",
            "lab3": "/predict/lab3-kmeans-clustering",
            "lab4": "/predict/lab4-random-forest",
            "lab5": "/predict/lab5-isolation-forest",
            "xgboost": "/predict/xgboost-classifier",
            "linear": "/predict/linear-regressor"
        },
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

