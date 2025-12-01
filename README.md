# MLflow Model Evaluation Lab — Comprehensive Documentation (Medium-Length Edition)

## 📘 1. Introduction
This project demonstrates how to evaluate Machine Learning models using **MLflow**, covering:
- **9 Machine Learning Models**: Logistic Regression, Decision Tree Regressor, K-Means Clustering, Random Forest Classifier, Isolation Forest, Wine Quality (Autologging), Wine Quality (Manual Logging), XGBoost Classifier, and Linear Regressor
- **MLflow Tracking**: Complete experiment tracking with PostgreSQL backend
- **FastAPI Prediction API**: RESTful endpoints for real-time model predictions
- **Docker Deployment**: Fully containerized setup with Docker Compose
- **Model Evaluation**: Automated metrics calculation and artifact generation
- **Production-Ready**: Health checks, lazy loading, and error handling

This guide is intentionally written in a **clear, structured, educational** style suitable for students, ML engineers, and DevOps/MLOps practitioners.

---

## 📂 2. Project Structure

```
mlflow-eval/
│
├── # Lab Scripts (9 ML Models)
├── lab1_logistic_regression.py      # Logistic Regression (Breast Cancer)
├── lab2_decision_tree_regressor.py  # Decision Tree Regressor (California Housing)
├── lab3_kmeans_clustering.py        # K-Means Clustering (Iris)
├── lab4_random_forest_classifier.py # Random Forest (Iris)
├── lab5_isolation_forest.py         # Isolation Forest (Breast Cancer)
├── lab6_wine_quality_autologging.py # Wine Quality (Autologging)
├── lab7_wine_quality_manual.py      # Wine Quality (Manual Logging)
├── classification.py                # XGBoost Classifier (Adult Dataset)
├── regressor.py                     # Linear Regressor (California Housing)
│
├── # Dataset Loader
├── dataset_loader.py                # Kaggle/UCI dataset download helper
│
├── # API Server
├── api_server.py                    # FastAPI prediction endpoints
│
├── # Docker Configuration
├── Dockerfile                       # Docker image for Python app
├── Dockerfile.mlflow-server        # Docker image for MLflow server
├── Dockerfile.api                   # Docker image for FastAPI server
├── docker-compose.yml               # Docker Compose configuration
├── entrypoint-mlflow.sh             # MLflow server startup script
│
├── # Documentation
├── README.md                        # Main documentation (this file)
├── MEDIUM.md                        # Comprehensive guide with theory & diagrams
├── POSTMAN_EXAMPLES.md              # Postman API testing guide
├── API_SUMMARY.md                   # Quick API reference
│
├── # Configuration
├── requirements.txt                 # Python dependencies
├── pip.conf                         # Pip configuration for timeouts
├── run_all_experiments.sh           # Run all experiments (Linux/Mac)
├── run_all_experiments.bat          # Run all experiments (Windows)
└── test_api.py                      # API testing script
```

---

## 🧠 3. MLflow Overview

MLflow is an open-source platform to manage the end-to-end ML lifecycle.

### **MLflow has four main components:**

```
+---------------------+
| MLflow Tracking     |
| - Log params        |
| - Log metrics       |
| - Log artifacts     |
| - Manage experiments|
+---------------------+

+---------------------+
| MLflow Models       |
| - Store models      |
| - Load models       |
| - Deploy models     |
+---------------------+

+---------------------+
| Model Registry      |
| - Staging/Prod      |
| - Versioning        |
+---------------------+

+---------------------+
| MLflow Projects     |
| (Not used here)     |
+---------------------+
```

### **Architecture Diagram (Simplified)**

```
                     +------------------------------+
                     |      MLflow Tracking UI      |
                     |    (http://localhost:5000)   |
                     +---------------+--------------+
                                     |
                                     v
                     +------------------------------+
                     |     MLflow Tracking Server   |
                     |  (Runs on port 5000)         |
                     +------+---------------+-------+
                            |               |
          Backend Store --->|               |<--- Artifact Store
                    (PostgreSQL)            (Docker Volume: mlruns/)
                            |               |
                            |               |
                     +------+---------------+-------+
                     |      FastAPI Server          |
                     |  (Runs on port 8000)        |
                     |  Prediction Endpoints       |
                     +------------------------------+
```

---

## ⚙️ 4. Installing Dependencies

### Install Python + pip + venv:
```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv
```

### Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

### Install project dependencies:
```bash
pip install mlflow scikit-learn xgboost shap
```

---

## 🐳 5. Docker Setup (Recommended)

This project includes Docker and Docker Compose configuration for easy setup and deployment. Using Docker ensures consistent environments across different machines and eliminates dependency conflicts.

### **Prerequisites**
- Docker Engine (version 20.10 or later)
- Docker Compose (version 2.0 or later)

### **Docker Architecture**

The Docker setup consists of four services working together:

```
+------------------+    +------------------+    +------------------+    +------------------+
|   PostgreSQL     |    |  mlflow-server   |    |   mlflow-app     |    |   mlflow-api     |
|   (Port 5432)    |<---|  (Port 5000)      |<---|  (Python App)    |    |  (Port 8000)     |
+------------------+    +------------------+    +------------------+    +------------------+
      |                        |                        |                        |
      v                        v                        v                        v
+------------------+    +------------------+    +------------------+    +------------------+
|  postgres-data   |    | mlflow-artifacts |    |  Project Files   |    |  Model Cache     |
|  (Volume)        |    |    (Volume)      |    |   (Bind Mount)    |    |  (In-Memory)     |
+------------------+    +------------------+    +------------------+    +------------------+
```

**Service Communication Flow:**
1. **PostgreSQL** stores all MLflow metadata (experiments, runs, metrics, parameters)
2. **MLflow Server** reads/writes to PostgreSQL and serves the UI on port 5000
3. **MLflow App** runs evaluation scripts and sends data to MLflow Server via HTTP
4. **MLflow API** loads models from MLflow Server and serves prediction endpoints on port 8000

### **How Components Work Together**

#### **1. docker-compose.yml - Orchestration Layer**

This file defines the entire stack:

- **PostgreSQL Service**: 
  - Uses `postgres:15-alpine` image
  - Creates database `mlflow` with user `mlflow`
  - Stores data in `postgres-data` volume (persists between restarts)
  - Health check ensures database is ready before other services start

- **MLflow Server Service**:
  - Built from `Dockerfile.mlflow-server`
  - Depends on PostgreSQL being healthy
  - Uses `entrypoint-mlflow.sh` as startup script
  - Receives PostgreSQL connection details via environment variables
  - Stores artifacts in `mlflow-artifacts` volume

- **MLflow App Service**:
  - Built from `Dockerfile` (contains Python dependencies)
  - Depends on MLflow Server being healthy
  - Mounts project directory for access to Python scripts
  - Receives MLflow server URI via environment variable

- **MLflow API Service** (FastAPI):
  - Built from `Dockerfile.api` (contains FastAPI and model dependencies)
  - Depends on MLflow Server being healthy
  - Loads models from MLflow experiments on startup (background loading, non-blocking)
  - Provides RESTful prediction endpoints on port 8000
  - Implements lazy loading for models not yet loaded
  - Shares `mlflow-artifacts` volume with MLflow Server for model access
  - Health check allows 10 minutes (600s) for model loading
  - Uses CORS middleware for cross-origin requests

#### **2. Dockerfile.mlflow-server - Server Image**

```dockerfile
FROM python:3.11-slim
# Installs: curl (for health checks), postgresql-client (for pg_isready)
# Installs: mlflow, psycopg2-binary (PostgreSQL adapter)
# Copies: entrypoint-mlflow.sh
# Sets: entrypoint to run the script on container start
```

**Key Components:**
- `postgresql-client`: Provides `pg_isready` command to check database availability
- `psycopg2-binary`: Python library to connect to PostgreSQL
- `entrypoint-mlflow.sh`: Custom startup script (see below)

#### **3. entrypoint-mlflow.sh - Server Startup Logic**

This script orchestrates the MLflow server startup:

**Step-by-step process:**

1. **Create directories**: Ensures `/app/mlruns` exists for artifacts
2. **Read environment variables**: Gets PostgreSQL connection details from docker-compose.yml
3. **Wait for PostgreSQL**: Uses `pg_isready` to wait until database is ready
4. **Construct connection URI**: Builds PostgreSQL connection string
5. **Start MLflow server**: Launches server with PostgreSQL as backend store

**Why this approach?**
- **Dependency management**: Ensures PostgreSQL is ready before MLflow starts
- **Configuration flexibility**: Uses environment variables (can be changed without rebuilding)
- **Error handling**: Waits for database instead of failing immediately
- **Logging**: Provides clear feedback about startup process

#### **4. Dockerfile - Application Image**

```dockerfile
FROM python:3.11-slim
# Installs: gcc, g++ (for compiling Python packages)
# Installs: All dependencies from requirements.txt
# Copies: Python scripts (classification.py, regressor.py)
# Sets: MLflow tracking URI via environment variable
```

**Purpose**: Provides isolated environment for running evaluation scripts with all dependencies pre-installed.

### **Data Flow Example**

When you run `docker-compose exec mlflow-app python classification.py`:

1. **Script execution**: `classification.py` runs in `mlflow-app` container
2. **Model training**: XGBoost trains on Adult dataset
3. **MLflow connection**: Script connects to `http://mlflow-server:5000` (via Docker network)
4. **Data logging**: MLflow server receives:
   - Metrics → Stored in PostgreSQL
   - Artifacts (models, plots) → Stored in `mlflow-artifacts` volume
5. **UI update**: MLflow UI reads from PostgreSQL and displays results

### **Volume Management**

- **postgres-data**: Persistent storage for all MLflow metadata
  - Experiments, runs, metrics, parameters
  - Survives container restarts
  - Located in Docker's volume directory

- **mlflow-artifacts**: Persistent storage for model files and plots
  - Saved models, confusion matrices, ROC curves, SHAP plots
  - Shared between `mlflow-server`, `mlflow-app`, and `mlflow-api`
  - Survives container restarts

### **Build and Run with Docker Compose**

#### **Step 1: Build the Docker Images**

```bash
docker-compose build
```

This command will:
- Build the `mlflow-app` image with all Python dependencies
- Build the `mlflow-server` image with MLflow and PostgreSQL client
- Build the `mlflow-api` image with FastAPI and model dependencies

#### **Step 2: Start the Services**

```bash
docker-compose up -d
```

This command will:
- Start PostgreSQL database on port 5432
- Start the MLflow tracking server on port 5000
- Start the Python application container (for running evaluation scripts)
- Start the FastAPI prediction server on port 8000
- Create shared volumes for PostgreSQL data and MLflow artifacts
- Wait for services to be healthy before starting dependent services

#### **Step 3: Verify Services are Running**

```bash
docker-compose ps
```

You should see all services running:
- `postgres` (container: `mlflow-postgres`, healthy)
- `mlflow-server` (container: `mlflow-server`, healthy)
- `mlflow-app` (container: `mlflow-app`, running)
- `mlflow-api` (container: `mlflow-api`, healthy)

**Note:** The `mlflow-api` health check allows 10 minutes for model loading on startup.

#### **Step 4: Access MLflow UI**

Open your browser and navigate to:
```
http://localhost:5000
```

You should see the MLflow Tracking UI with no experiments yet.

#### **Step 5: Access FastAPI Prediction API**

Open your browser and navigate to:
```
http://localhost:8000/docs
```

You should see the interactive Swagger UI for the prediction API. The API provides endpoints for all 7 trained models.

**Health Check:**
```
http://localhost:8000/health
```

**API Root:**
```
http://localhost:8000/
```

### **Running Evaluation Scripts**

#### **Option 1: Execute Scripts in Container**

Run individual experiments:
```bash
# Lab 1: Logistic Regression
docker-compose exec mlflow-app python lab1_logistic_regression.py

# Lab 2: Decision Tree Regressor
docker-compose exec mlflow-app python lab2_decision_tree_regressor.py

# Lab 3: K-Means Clustering
docker-compose exec mlflow-app python lab3_kmeans_clustering.py

# Lab 4: Random Forest Classifier
docker-compose exec mlflow-app python lab4_random_forest_classifier.py

# Lab 5: Isolation Forest
docker-compose exec mlflow-app python lab5_isolation_forest.py

# XGBoost Classifier
docker-compose exec mlflow-app python classification.py

# Linear Regressor
docker-compose exec mlflow-app python regressor.py
```

#### **Option 1b: Run All Experiments**

On Linux/Mac:
```bash
docker-compose exec mlflow-app bash -c "./run_all_experiments.sh"
```

On Windows:
```bash
docker-compose exec mlflow-app bash -c "./run_all_experiments.bat"
```

#### **Option 2: Run Scripts Interactively**

Open an interactive shell in the container:
```bash
docker-compose exec mlflow-app bash
```

Then run scripts:
```bash
python classification.py
python regressor.py
```

### **Viewing Logs**

View logs from all services:
```bash
docker-compose logs -f
```

View logs from a specific service:
```bash
docker-compose logs -f mlflow-server
docker-compose logs -f mlflow-app
docker-compose logs -f mlflow-api
docker-compose logs -f postgres
```

### **Stopping Services**

Stop all services:
```bash
docker-compose down
```

Stop and remove volumes (⚠️ **WARNING**: This deletes all MLflow data):
```bash
docker-compose down -v
```

### **Rebuilding After Changes**

If you modify `requirements.txt` or `Dockerfile`:
```bash
docker-compose build --no-cache
docker-compose up -d
```

### **Docker Configuration Details**

- **PostgreSQL**: Runs on port 5432, accessible from host machine
- **MLflow Server**: Runs on port 5000, accessible from host machine
- **Backend Store**: PostgreSQL database (persistent in `postgres-data` volume)
- **Artifact Store**: `mlruns/` directory (persistent in `mlflow-artifacts` volume)
- **Networking**: Services communicate via Docker's internal network
- **Environment Variables**: Connection details passed via docker-compose.yml
- **Health Checks**: Ensures services start in correct order (PostgreSQL → MLflow Server → MLflow App & MLflow API)
- **Service Names**: 
  - `postgres` (container: `mlflow-postgres`)
  - `mlflow-server` (container: `mlflow-server`)
  - `mlflow-app` (container: `mlflow-app`)
  - `mlflow-api` (container: `mlflow-api`)

---

## 🚀 6. MLflow Server Setup (Local Installation - Optional)

If you prefer to run MLflow locally without Docker, you'll need PostgreSQL installed:

```bash
# Install PostgreSQL (Ubuntu/Debian)
sudo apt update
sudo apt install -y postgresql postgresql-contrib

# Create database
sudo -u postgres createdb mlflow
sudo -u postgres createuser mlflow
sudo -u postgres psql -c "ALTER USER mlflow WITH PASSWORD 'mlflow';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow;"

# Start MLflow server
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri postgresql://mlflow:mlflow@localhost:5432/mlflow \
  --artifacts-destination ./mlruns \
  --serve-artifacts \
  --allowed-hosts '*' \
  --cors-allowed-origins '*'
```

**Note**: Docker setup is recommended as it handles all dependencies automatically.

---

## 🧪 7. Classification Model Evaluation (XGBoost)

### **Dataset**
We use the *UCI Adult* dataset loaded via SHAP:
- Predicts whether a person earns >50k  
- Contains categorical + numerical features  

### **Model**
We train a simple **XGBoostClassifier**.

### **Evaluation**
MLflow automatically generates:
- Accuracy  
- F1 Score  
- ROC-AUC  
- Confusion matrix  
- ROC curve  
- SHAP summary plot  

### **classification.py - Code Explanation**

```python
import os
import mlflow
import mlflow.sklearn
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from mlflow.models import infer_signature
from mlflow.models import evaluate as mlflow_evaluate

# Connect to MLflow server (uses environment variable in Docker)
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_registry_uri(tracking_uri)

# Create or get experiment
experiment_name = "evaluation"
mlflow.set_experiment(experiment_name)

# Load and split data
X, y = shap.datasets.adult()  # UCI Adult dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train model
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Prepare evaluation data
eval_data = X_test.copy()
eval_data["label"] = y_test  # Add target column for MLflow evaluation

# Start MLflow run
with mlflow.start_run(run_name="xgboost_adult_classifier"):
    # Generate predictions for signature inference
    y_pred = model.predict(X_test)
    signature = infer_signature(X_test, y_pred)  # Auto-detect input/output schema

    # Log model to MLflow
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
    )

    # Define prediction function for evaluation
    def predict_fn(df):
        features = df.drop(columns=["label"], errors="ignore")
        return model.predict(features)

    # Evaluate model using MLflow's built-in evaluator
    result = mlflow_evaluate(
        model=predict_fn,           # Prediction function
        data=eval_data,              # Evaluation dataset with targets
        targets="label",             # Target column name
        model_type="classifier",     # Model type (classifier/regressor)
        evaluators=["default"],      # Use default evaluator
    )

    # Print metrics (with error handling)
    if 'accuracy_score' in result.metrics:
        print(f"Accuracy: {result.metrics['accuracy_score']:.3f}")
    if 'f1_score' in result.metrics:
        print(f"F1 Score: {result.metrics['f1_score']:.3f}")
    if 'roc_auc_score' in result.metrics:
        print(f"ROC AUC: {result.metrics['roc_auc_score']:.3f}")
    
    # Show all available metrics
    print(f"\nAll available metrics: {list(result.metrics.keys())}")
```

**Key Components Explained:**

1. **Environment Variable**: `MLFLOW_TRACKING_URI` is set by docker-compose.yml to `http://mlflow-server:5000`
2. **Experiment Management**: Creates/uses experiment named "evaluation"
3. **Model Logging**: Saves trained model with signature (input/output schema)
4. **Model Evaluation**: Uses MLflow's evaluator to automatically calculate metrics and generate plots
5. **Artifacts**: MLflow automatically generates confusion matrix, ROC curve, and SHAP plots

---

## 📈 8. Regression Model Evaluation (Linear Regression)

### **Dataset**
California Housing dataset:
- Predicts median house value  
- Contains numeric features  

### **Evaluation Metrics**
- MAE — Mean Absolute Error  
- RMSE — Root Mean Squared Error  
- R² — Coefficient of Determination  

### **regressor.py - Code Explanation**

```python
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from mlflow.models import infer_signature
from mlflow.models import evaluate as mlflow_evaluate
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

# Connect to MLflow server
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_registry_uri(tracking_uri)

# Create separate experiment for regression
experiment_name = "evaluation2"
mlflow.set_experiment(experiment_name)

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, test_size=0.2, random_state=42
)

# Train linear regression model
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# Prepare evaluation data
eval_data = X_test.copy()
eval_data["target"] = y_test

# Start MLflow run
with mlflow.start_run(run_name="linreg_california_housing"):
    # Generate predictions for signature
    y_pred = reg_model.predict(X_train)
    signature = infer_signature(X_train, y_pred)

    # Log model
    mlflow.sklearn.log_model(
        sk_model=reg_model,
        artifact_path="model",
        signature=signature,
    )

    # Prediction function for evaluation
    def predict_fn(df):
        features = df.drop(columns=["target"], errors="ignore")
        return reg_model.predict(features)

    # Evaluate regression model
    result = mlflow_evaluate(
        model=predict_fn,
        data=eval_data,
        targets="target",
        model_type="regressor",      # Different model type
        evaluators=["default"],
    )

    # Print regression metrics (with error handling)
    if 'mean_absolute_error' in result.metrics:
        print(f"MAE:  {result.metrics['mean_absolute_error']:.3f}")
    if 'root_mean_squared_error' in result.metrics:
        print(f"RMSE: {result.metrics['root_mean_squared_error']:.3f}")
    if 'r2_score' in result.metrics:
        print(f"R²:   {result.metrics['r2_score']:.3f}")
    
    # Show all available metrics
    print(f"\nAll available metrics: {list(result.metrics.keys())}")
```

**Differences from Classification:**
- Uses `model_type="regressor"` instead of `"classifier"`
- Calculates regression-specific metrics (MAE, RMSE, R²)
- Uses California Housing dataset (continuous target variable)

---

## 📊 9. Evaluation Metrics — Theory Summary

### **Classification Metrics**

| Metric | Meaning |
|-------|---------|
| **Accuracy** | % of correct predictions |
| **F1 Score** | Balance between precision and recall |
| **ROC-AUC** | Ability to separate classes across thresholds |

### **Regression Metrics**

| Metric | Meaning |
|--------|---------|
| **MAE** | Average absolute error |
| **RMSE** | Penalizes large errors more heavily |
| **R²** | Explained variance (1 = perfect) |

---

## 🧩 10. SHAP Explainability (Short Summary)

SHAP explains how each feature contributes to predictions.

ASCII diagram:

```
           +------------------------+
           | SHAP Explainer         |
           +----------+-------------+
                      |
                      v
           +------------------------+
           | Feature Contributions  |
           | e.g. age +3.1, hours  -1.2
           +------------------------+
```

---

## 🛠 11. Common Issues & Fixes

### ❌ **500 Internal Server Error on artifacts endpoint**
Cause: MLflow not serving artifacts  
Fix: Use `--serve-artifacts` flag (already in entrypoint-mlflow.sh)

### ❌ **PostgreSQL connection errors**
Cause: MLflow server starting before PostgreSQL is ready  
Fix: `entrypoint-mlflow.sh` waits for PostgreSQL using `pg_isready` (already implemented)

### ❌ **KeyError: 'roc_auc_score' or other metrics**
Cause: Some metrics may not be available depending on evaluation configuration  
Fix: Scripts now check if metrics exist before accessing (already fixed)

### ❌ **Model evaluation failing with URI**
Cause: MLflow expects PyFunc or function for local evaluation  
Fix: Use `predict_fn` as shown in scripts (already implemented)

### ❌ **Experiment not found**
Cause: Opening old experiment link after switching backend DB  
Fix: Use new experiment links in UI after switching databases

### ❌ **Docker volume permission issues (Windows)**
Cause: Windows Docker has issues with SQLite on mounted volumes  
Fix: Using PostgreSQL with Docker volumes (already implemented) - this resolves the issue

---

## 🧪 12. Testing the Project

This section describes how to test and verify that the MLflow evaluation lab is working correctly.

### **Test Prerequisites**

Before running tests, ensure:
- Docker services are running (`docker-compose ps` shows both services)
- MLflow UI is accessible at `http://localhost:5000`
- No errors in container logs (`docker-compose logs`)

### **Test 1: MLflow Server Health Check**

Verify the MLflow server is responding:

```bash
# Check server health endpoint
curl http://localhost:5000/health

# Expected output: {"status": "ok"}
```

Or open in browser: `http://localhost:5000/health`

### **Test 2: Classification Model Evaluation**

Run the classification script and verify output:

```bash
docker-compose exec mlflow-app python classification.py
```

**Expected Output:**
```
Accuracy: 0.XXX
F1 Score: 0.XXX
ROC AUC: 0.XXX
```

**What to Verify:**
1. ✅ Script runs without errors
2. ✅ Metrics are printed (Accuracy, F1 Score, ROC AUC)
3. ✅ MLflow UI shows new experiment "evaluation"
4. ✅ MLflow UI shows run "xgboost_adult_classifier"
5. ✅ Metrics are visible in MLflow UI
6. ✅ Artifacts (confusion matrix, ROC curve, SHAP plots) are logged

**Verify in MLflow UI:**
- Navigate to `http://localhost:5000`
- Click on experiment "evaluation"
- Click on run "xgboost_adult_classifier"
- Check **Metrics** tab: Should show `accuracy_score`, `f1_score`, `roc_auc_score`
- Check **Artifacts** tab: Should show `model/`, `confusion_matrix.png`, `roc_curve.png`, etc.

### **Test 3: Regression Model Evaluation**

Run the regression script and verify output:

```bash
docker-compose exec mlflow-app python regressor.py
```

**Expected Output:**
```
MAE:  X.XXX
RMSE: X.XXX
R²:   X.XXX
```

**What to Verify:**
1. ✅ Script runs without errors
2. ✅ Metrics are printed (MAE, RMSE, R²)
3. ✅ MLflow UI shows new experiment "evaluation2"
4. ✅ MLflow UI shows run "linreg_california_housing"
5. ✅ Metrics are visible in MLflow UI
6. ✅ Model artifacts are logged

**Verify in MLflow UI:**
- Navigate to `http://localhost:5000`
- Click on experiment "evaluation2"
- Click on run "linreg_california_housing"
- Check **Metrics** tab: Should show `mean_absolute_error`, `root_mean_squared_error`, `r2_score`
- Check **Artifacts** tab: Should show `model/` directory

### **Test 4: Data Persistence**

Verify that MLflow data persists across container restarts:

```bash
# Stop containers
docker-compose down

# Start containers again
docker-compose up -d

# Verify experiments still exist in UI
# Open http://localhost:5000
```

**What to Verify:**
1. ✅ Previous experiments are still visible
2. ✅ Previous runs are still accessible
3. ✅ Artifacts are still available
4. ✅ Metrics are still displayed

### **Test 5: Multiple Runs**

Run scripts multiple times to verify multiple runs are tracked:

```bash
# Run classification 3 times
docker-compose exec mlflow-app python classification.py
docker-compose exec mlflow-app python classification.py
docker-compose exec mlflow-app python classification.py
```

**What to Verify:**
1. ✅ Each run creates a new entry in MLflow UI
2. ✅ All runs are visible under the same experiment
3. ✅ Metrics can be compared across runs
4. ✅ Each run has unique timestamps

### **Test 6: Container Connectivity**

Verify containers can communicate:

```bash
# Test network connectivity from app to server
docker-compose exec mlflow-app curl http://mlflow-server:5000/health
```

**Expected Output:**
```
{"status": "ok"}
```

### **Test 7: PostgreSQL Connection**

Verify PostgreSQL is working:

```bash
# Check PostgreSQL container
docker-compose exec postgres psql -U mlflow -d mlflow -c "SELECT COUNT(*) FROM experiments;"

# Check MLflow can access PostgreSQL
docker-compose logs mlflow-server | grep "PostgreSQL is ready"
```

**What to Verify:**
1. ✅ PostgreSQL container is running
2. ✅ MLflow server connected to PostgreSQL
3. ✅ Experiments table exists in database

### **Troubleshooting Tests**

If any test fails:

1. **Check container logs:**
   ```bash
   docker-compose logs mlflow-server
   docker-compose logs mlflow-app
   ```

2. **Check container status:**
   ```bash
   docker-compose ps
   ```

3. **Restart services:**
   ```bash
   docker-compose restart
   ```

4. **Rebuild containers:**
   ```bash
   docker-compose down
   docker-compose build --no-cache
   docker-compose up -d
   ```

### **Test Summary Checklist**

- [ ] PostgreSQL container is running and healthy
- [ ] MLflow server is accessible at `http://localhost:5000`
- [ ] MLflow server connected to PostgreSQL successfully
- [ ] Classification script runs successfully
- [ ] Regression script runs successfully
- [ ] Metrics are logged to MLflow UI
- [ ] Artifacts are visible in MLflow UI
- [ ] Data persists after container restart (PostgreSQL + volumes)
- [ ] Multiple runs are tracked correctly
- [ ] Containers can communicate via Docker network

---

## 🎯 13. Conclusion

You now have:
- **Fully containerized MLflow setup** with Docker Compose
- **PostgreSQL backend** for reliable data persistence (works on Windows, Linux, macOS)
- **7 Machine Learning models** covering classification, regression, clustering, and anomaly detection
- **FastAPI prediction API** with RESTful endpoints for real-time predictions
- **Proper MLflow server configuration** with health checks and dependency management
- **SHAP-based explainability** for model interpretation
- **Automatic metrics & artifact logging** to MLflow UI
- **Lazy model loading** for efficient resource usage
- **Clean, reproducible ML evaluation workflow** with isolated environments

### **Architecture Benefits**

- **Scalability**: PostgreSQL can handle large-scale ML experiments
- **Reliability**: Health checks ensure services start in correct order
- **Portability**: Works identically on Windows, Linux, and macOS
- **Persistence**: Data survives container restarts via Docker volumes
- **Isolation**: Each service runs in its own container with specific dependencies

### **Production Readiness**

This setup mirrors **real-world MLOps workflows** and prepares you for:
- **Model Registry**: Track model versions and stages
- **Deployment pipelines**: Integrate with CI/CD systems
- **Cloud ML workflows**: Easy migration to cloud services (AWS, GCP, Azure)
- **Team collaboration**: Shared PostgreSQL database for team access
- **Monitoring**: MLflow UI provides experiment tracking and comparison

### **Next Steps**

1. **Explore MLflow UI**: Navigate to `http://localhost:5000` and explore experiments
2. **Run multiple experiments**: Modify hyperparameters and compare results
3. **Export models**: Download models from MLflow UI for deployment
4. **Customize evaluation**: Add custom metrics or evaluators
5. **Scale up**: Consider adding more services (model serving, monitoring)

Happy experimenting 🚀


