"""
Simple test script to verify API endpoints are working
Run this after training all models and starting the API server
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_lab1():
    """Test Lab 1: Logistic Regression"""
    print("Testing Lab 1: Logistic Regression...")
    data = {
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
    response = requests.post(f"{BASE_URL}/predict/lab1-logistic-regression", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_lab2():
    """Test Lab 2: Decision Tree Regressor"""
    print("Testing Lab 2: Decision Tree Regressor...")
    data = {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.98412698,
        "AveBedrms": 1.02380952,
        "Population": 322.0,
        "AveOccup": 2.55555556,
        "Latitude": 37.88,
        "Longitude": -122.23
    }
    response = requests.post(f"{BASE_URL}/predict/lab2-decision-tree-regressor", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_lab3():
    """Test Lab 3: K-Means Clustering"""
    print("Testing Lab 3: K-Means Clustering...")
    data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = requests.post(f"{BASE_URL}/predict/lab3-kmeans-clustering", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_lab4():
    """Test Lab 4: Random Forest"""
    print("Testing Lab 4: Random Forest Classifier...")
    data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = requests.post(f"{BASE_URL}/predict/lab4-random-forest", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

if __name__ == "__main__":
    print("=" * 50)
    print("MLflow API Test Suite")
    print("=" * 50)
    print()
    
    try:
        test_health()
        test_lab1()
        test_lab2()
        test_lab3()
        test_lab4()
        
        print("=" * 50)
        print("Tests completed!")
        print("=" * 50)
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to API server.")
        print("Make sure the API server is running:")
        print("  docker-compose up -d mlflow-api")
    except Exception as e:
        print(f"ERROR: {str(e)}")

