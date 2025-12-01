# Postman API Testing Guide

This document provides Postman collection examples for testing all prediction endpoints.

## Base URL
```
http://localhost:8000
```

## API Documentation
Interactive API docs available at: `http://localhost:8000/docs`

---

## 1. Health Check

**GET** `/health`

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": 9,
  "models": [
    "lab1_logistic_regression",
    "lab2_decision_tree_regressor",
    "lab3_kmeans_clustering",
    "lab4_random_forest",
    "lab5_isolation_forest",
    "lab6_wine_quality_autolog",
    "lab7_wine_quality_manual",
    "xgboost_classifier",
    "linear_regressor"
  ]
}
```

---

## 2. Lab 1: Logistic Regression (Breast Cancer Classification)

**POST** `/predict/lab1-logistic-regression`

**Request Body (JSON):**
```json
{
  "mean_radius": 17.99,
  "mean_texture": 10.38,
  "mean_perimeter": 122.8,
  "mean_area": 1001.0,
  "mean_smoothness": 0.1184,
  "mean_compactness": 0.2776,
  "mean_concavity": 0.3001,
  "mean_concave_points": 0.1471,
  "mean_symmetry": 0.2419,
  "mean_fractal_dimension": 0.07871,
  "radius_error": 1.095,
  "texture_error": 0.9053,
  "perimeter_error": 8.589,
  "area_error": 153.4,
  "smoothness_error": 0.006399,
  "compactness_error": 0.04904,
  "concavity_error": 0.05373,
  "concave_points_error": 0.01587,
  "symmetry_error": 0.03003,
  "fractal_dimension_error": 0.006193,
  "worst_radius": 25.38,
  "worst_texture": 17.33,
  "worst_perimeter": 184.6,
  "worst_area": 2019.0,
  "worst_smoothness": 0.1622,
  "worst_compactness": 0.6656,
  "worst_concavity": 0.7119,
  "worst_concave_points": 0.2654,
  "worst_symmetry": 0.4601,
  "worst_fractal_dimension": 0.1189
}
```

**Response:**
```json
{
  "prediction": 1,
  "prediction_label": "benign",
  "prediction_probability": {
    "malignant": 0.023,
    "benign": 0.977
  },
  "model_name": "lab1-logistic-regression",
  "prediction_type": "classification"
}
```

---

## 3. Lab 2: Decision Tree Regressor (California Housing)

**POST** `/predict/lab2-decision-tree-regressor`

**Request Body (JSON):**
```json
{
  "MedInc": 8.3252,
  "HouseAge": 41.0,
  "AveRooms": 6.98412698,
  "AveBedrms": 1.02380952,
  "Population": 322.0,
  "AveOccup": 2.55555556,
  "Latitude": 37.88,
  "Longitude": -122.23
}
```

**Response:**
```json
{
  "prediction": 4.526,
  "model_name": "lab2-decision-tree-regressor",
  "prediction_type": "regression",
  "unit": "median house value (in hundreds of thousands)"
}
```

---

## 4. Lab 3: K-Means Clustering (Iris)

**POST** `/predict/lab3-kmeans-clustering`

**Request Body (JSON):**
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Response:**
```json
{
  "prediction": 0,
  "cluster_label": "cluster_0",
  "model_name": "lab3-kmeans-clustering",
  "prediction_type": "clustering"
}
```

---

## 5. Lab 4: Random Forest Classifier (Iris)

**POST** `/predict/lab4-random-forest`

**Request Body (JSON):**
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Response:**
```json
{
  "prediction": 0,
  "prediction_label": "setosa",
  "prediction_probability": {
    "setosa": 0.98,
    "versicolor": 0.02,
    "virginica": 0.00
  },
  "model_name": "lab4-random-forest",
  "prediction_type": "classification"
}
```

---

## 6. Lab 5: Isolation Forest (Anomaly Detection)

**POST** `/predict/lab5-isolation-forest`

**Request Body (JSON):**
```json
{
  "mean_radius": 17.99,
  "mean_texture": 10.38,
  "mean_perimeter": 122.8,
  "mean_area": 1001.0,
  "mean_smoothness": 0.1184,
  "mean_compactness": 0.2776,
  "mean_concavity": 0.3001,
  "mean_concave_points": 0.1471,
  "mean_symmetry": 0.2419,
  "mean_fractal_dimension": 0.07871,
  "radius_error": 1.095,
  "texture_error": 0.9053,
  "perimeter_error": 8.589,
  "area_error": 153.4,
  "smoothness_error": 0.006399,
  "compactness_error": 0.04904,
  "concavity_error": 0.05373,
  "concave_points_error": 0.01587,
  "symmetry_error": 0.03003,
  "fractal_dimension_error": 0.006193,
  "worst_radius": 25.38,
  "worst_texture": 17.33,
  "worst_perimeter": 184.6,
  "worst_area": 2019.0,
  "worst_smoothness": 0.1622,
  "worst_compactness": 0.6656,
  "worst_concavity": 0.7119,
  "worst_concave_points": 0.2654,
  "worst_symmetry": 0.4601,
  "worst_fractal_dimension": 0.1189
}
```

**Response:**
```json
{
  "prediction": 1,
  "prediction_label": "normal",
  "model_name": "lab5-isolation-forest",
  "prediction_type": "anomaly_detection"
}
```

---

## 7. Lab 6: Wine Quality Classification (Autologging)

**POST** `/predict/lab6-wine-quality-autologging`

**Request Body (JSON):**
```json
{
  "fixed_acidity": 7.4,
  "volatile_acidity": 0.7,
  "citric_acid": 0.0,
  "residual_sugar": 1.9,
  "chlorides": 0.076,
  "free_sulfur_dioxide": 11.0,
  "total_sulfur_dioxide": 34.0,
  "density": 0.9978,
  "pH": 3.51,
  "sulphates": 0.56,
  "alcohol": 9.4
}
```

**Response:**
```json
{
  "prediction": 0,
  "prediction_label": "bad",
  "prediction_probability": {
    "bad": 0.85,
    "good": 0.15
  },
  "model_name": "lab6-wine-quality-autologging",
  "prediction_type": "classification"
}
```

---

## 8. Lab 7: Wine Quality Classification (Manual Logging)

**POST** `/predict/lab7-wine-quality-manual`

**Request Body (JSON):**
```json
{
  "fixed_acidity": 7.4,
  "volatile_acidity": 0.7,
  "citric_acid": 0.0,
  "residual_sugar": 1.9,
  "chlorides": 0.076,
  "free_sulfur_dioxide": 11.0,
  "total_sulfur_dioxide": 34.0,
  "density": 0.9978,
  "pH": 3.51,
  "sulphates": 0.56,
  "alcohol": 9.4
}
```

**Response:**
```json
{
  "prediction": 0,
  "prediction_label": "bad",
  "prediction_probability": {
    "bad": 0.85,
    "good": 0.15
  },
  "model_name": "lab7-wine-quality-manual",
  "prediction_type": "classification"
}
```

---

## 9. XGBoost Classifier (Adult Dataset)

**POST** `/predict/xgboost-classifier`

**Request Body (JSON):**
```json
{
  "Age": 39,
  "Workclass": "State-gov",
  "Fnlwgt": 77516,
  "Education": "Bachelors",
  "Education-Num": 13,
  "Marital-Status": "Never-married",
  "Occupation": "Adm-clerical",
  "Relationship": "Not-in-family",
  "Race": "White",
  "Sex": "Male",
  "Capital-Gain": 2174,
  "Capital-Loss": 0,
  "Hours-per-week": 40,
  "Native-Country": "United-States"
}
```

**Response:**
```json
{
  "prediction": 0,
  "prediction_label": "<=50K",
  "prediction_probability": {
    "<=50K": 0.85,
    ">50K": 0.15
  },
  "model_name": "xgboost-classifier",
  "prediction_type": "classification"
}
```

---

## 10. Linear Regressor (California Housing)

**POST** `/predict/linear-regressor`

**Request Body (JSON):**
```json
{
  "MedInc": 8.3252,
  "HouseAge": 41.0,
  "AveRooms": 6.98412698,
  "AveBedrms": 1.02380952,
  "Population": 322.0,
  "AveOccup": 2.55555556,
  "Latitude": 37.88,
  "Longitude": -122.23
}
```

**Response:**
```json
{
  "prediction": 4.526,
  "model_name": "linear-regressor",
  "prediction_type": "regression",
  "unit": "median house value (in hundreds of thousands)"
}
```

---

## Postman Collection Setup

### Import Collection

1. Open Postman
2. Click "Import"
3. Create a new collection named "MLflow Prediction API"
4. Add all endpoints above

### Environment Variables (Optional)

Create a Postman environment with:
- `base_url`: `http://localhost:8000`
- `mlflow_ui`: `http://localhost:5000`

### Testing Tips

1. **Start with Health Check**: Always test `/health` first to verify API is running
2. **Check Model Loading**: Verify all models are loaded in health response
3. **Test Each Endpoint**: Use the example JSON payloads above
4. **Error Handling**: Test with invalid data to see error responses
5. **Response Time**: Monitor response times for each endpoint

### Common Errors

**503 Service Unavailable**: Model not loaded - train the model first
**422 Validation Error**: Invalid input data format
**500 Internal Server Error**: Model prediction failed - check logs

---

## cURL Examples

### Lab 1 - Logistic Regression
```bash
curl -X POST "http://localhost:8000/predict/lab1-logistic-regression" \
  -H "Content-Type: application/json" \
  -d '{
    "mean_radius": 17.99,
    "mean_texture": 10.38,
    "mean_perimeter": 122.8,
    "mean_area": 1001.0,
    "mean_smoothness": 0.1184,
    "mean_compactness": 0.2776,
    "mean_concavity": 0.3001,
    "mean_concave_points": 0.1471,
    "mean_symmetry": 0.2419,
    "mean_fractal_dimension": 0.07871,
    "radius_error": 1.095,
    "texture_error": 0.9053,
    "perimeter_error": 8.589,
    "area_error": 153.4,
    "smoothness_error": 0.006399,
    "compactness_error": 0.04904,
    "concavity_error": 0.05373,
    "concave_points_error": 0.01587,
    "symmetry_error": 0.03003,
    "fractal_dimension_error": 0.006193,
    "worst_radius": 25.38,
    "worst_texture": 17.33,
    "worst_perimeter": 184.6,
    "worst_area": 2019.0,
    "worst_smoothness": 0.1622,
    "worst_compactness": 0.6656,
    "worst_concavity": 0.7119,
    "worst_concave_points": 0.2654,
    "worst_symmetry": 0.4601,
    "worst_fractal_dimension": 0.1189
  }'
```

### Lab 2 - Decision Tree Regressor
```bash
curl -X POST "http://localhost:8000/predict/lab2-decision-tree-regressor" \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.98412698,
    "AveBedrms": 1.02380952,
    "Population": 322.0,
    "AveOccup": 2.55555556,
    "Latitude": 37.88,
    "Longitude": -122.23
  }'
```

### Lab 3 - K-Means
```bash
curl -X POST "http://localhost:8000/predict/lab3-kmeans-clustering" \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

### Lab 4 - Random Forest
```bash
curl -X POST "http://localhost:8000/predict/lab4-random-forest" \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

### Lab 5 - Isolation Forest
```bash
curl -X POST "http://localhost:8000/predict/lab5-isolation-forest" \
  -H "Content-Type: application/json" \
  -d '{
    "mean_radius": 17.99,
    "mean_texture": 10.38,
    "mean_perimeter": 122.8,
    "mean_area": 1001.0,
    "mean_smoothness": 0.1184,
    "mean_compactness": 0.2776,
    "mean_concavity": 0.3001,
    "mean_concave_points": 0.1471,
    "mean_symmetry": 0.2419,
    "mean_fractal_dimension": 0.07871,
    "radius_error": 1.095,
    "texture_error": 0.9053,
    "perimeter_error": 8.589,
    "area_error": 153.4,
    "smoothness_error": 0.006399,
    "compactness_error": 0.04904,
    "concavity_error": 0.05373,
    "concave_points_error": 0.01587,
    "symmetry_error": 0.03003,
    "fractal_dimension_error": 0.006193,
    "worst_radius": 25.38,
    "worst_texture": 17.33,
    "worst_perimeter": 184.6,
    "worst_area": 2019.0,
    "worst_smoothness": 0.1622,
    "worst_compactness": 0.6656,
    "worst_concavity": 0.7119,
    "worst_concave_points": 0.2654,
    "worst_symmetry": 0.4601,
    "worst_fractal_dimension": 0.1189
  }'
```

### Lab 6 - Wine Quality Autologging
```bash
curl -X POST "http://localhost:8000/predict/lab6-wine-quality-autologging" \
  -H "Content-Type: application/json" \
  -d '{
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.7,
    "citric_acid": 0.0,
    "residual_sugar": 1.9,
    "chlorides": 0.076,
    "free_sulfur_dioxide": 11.0,
    "total_sulfur_dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4
  }'
```

### Lab 7 - Wine Quality Manual
```bash
curl -X POST "http://localhost:8000/predict/lab7-wine-quality-manual" \
  -H "Content-Type: application/json" \
  -d '{
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.7,
    "citric_acid": 0.0,
    "residual_sugar": 1.9,
    "chlorides": 0.076,
    "free_sulfur_dioxide": 11.0,
    "total_sulfur_dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4
  }'
```

### XGBoost Classifier
```bash
curl -X POST "http://localhost:8000/predict/xgboost-classifier" \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 39,
    "Workclass": "State-gov",
    "Fnlwgt": 77516,
    "Education": "Bachelors",
    "Education-Num": 13,
    "Marital-Status": "Never-married",
    "Occupation": "Adm-clerical",
    "Relationship": "Not-in-family",
    "Race": "White",
    "Sex": "Male",
    "Capital-Gain": 2174,
    "Capital-Loss": 0,
    "Hours-per-week": 40,
    "Native-Country": "United-States"
  }'
```

### Linear Regressor
```bash
curl -X POST "http://localhost:8000/predict/linear-regressor" \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.98412698,
    "AveBedrms": 1.02380952,
    "Population": 322.0,
    "AveOccup": 2.55555556,
    "Latitude": 37.88,
    "Longitude": -122.23
  }'
```

---

## Testing Workflow

1. **Start Services**: `docker-compose up -d`
2. **Train Models**: Run all training scripts first
   ```bash
   # Run all lab scripts including Lab 6 and Lab 7
   docker-compose exec mlflow-app python lab1_logistic_regression.py
   docker-compose exec mlflow-app python lab2_decision_tree_regressor.py
   docker-compose exec mlflow-app python lab3_kmeans_clustering.py
   docker-compose exec mlflow-app python lab4_random_forest_classifier.py
   docker-compose exec mlflow-app python lab5_isolation_forest.py
   docker-compose exec mlflow-app python lab6_wine_quality_autologging.py
   docker-compose exec mlflow-app python lab7_wine_quality_manual.py
   docker-compose exec mlflow-app python classification.py
   docker-compose exec mlflow-app python regressor.py
   ```
3. **Check Health**: `GET http://localhost:8000/health`
4. **Test Predictions**: Use Postman or cURL examples above
5. **View API Docs**: `http://localhost:8000/docs` (Swagger UI)

## Note on Lab 6 and Lab 7

**Lab 6 (Wine Quality Autologging)** and **Lab 7 (Wine Quality Manual Logging)** are evaluation scripts that demonstrate MLflow logging capabilities. They:

- Train Random Forest models on Wine Quality dataset
- Demonstrate autologging vs manual logging approaches
- Use Kaggle API for dataset download (with UCI fallback)
- Log comprehensive metrics and artifacts to MLflow
- **Both have API prediction endpoints** (see sections 7 and 8 above)

**To run these labs:**
```bash
# Lab 6: Autologging demonstration
docker-compose exec mlflow-app python lab6_wine_quality_autologging.py

# Lab 7: Manual logging demonstration
docker-compose exec mlflow-app python lab7_wine_quality_manual.py
```

**Kaggle Setup (Optional):**
- Create `.env` file with `KAGGLE_USERNAME` and `KAGGLE_KEY`
- See `KAGGLE_SETUP.md` for detailed instructions
- If not configured, scripts automatically use UCI ML Repository

**API Endpoints:**
- Lab 6: `POST /predict/lab6-wine-quality-autologging`
- Lab 7: `POST /predict/lab7-wine-quality-manual`

