"""
Automated tests for the Self-Healing ML Pipeline API.

Run with: pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path so we can import the app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from importlib import import_module
fastapi_module = import_module("05_fastapi_app")
app = fastapi_module.app

client = TestClient(app)


# ============================================
# TEST 1: Root endpoint
# ============================================
def test_root_endpoint():
    """Test that the root endpoint returns API info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "endpoints" in data
    assert "/predict" in data["endpoints"]


# ============================================
# TEST 2: Predict endpoint
# ============================================
def test_predict_setosa():
    """Test prediction returns setosa for typical setosa features."""
    response = client.get("/predict", params={
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    })
    assert response.status_code == 200
    data = response.json()
    assert data["flower"] == "setosa"
    assert data["prediction"] == 0


def test_predict_virginica():
    """Test prediction returns virginica for typical virginica features."""
    response = client.get("/predict", params={
        "sepal_length": 6.5,
        "sepal_width": 3.0,
        "petal_length": 5.5,
        "petal_width": 2.0
    })
    assert response.status_code == 200
    data = response.json()
    assert data["flower"] == "virginica"


def test_predict_missing_params():
    """Test that missing parameters return 422."""
    response = client.get("/predict", params={"sepal_length": 5.1})
    assert response.status_code == 422


# ============================================
# TEST 3: Drift report endpoint
# ============================================
def test_drift_report_structure():
    """Test drift report returns expected fields."""
    response = client.get("/drift-report")
    assert response.status_code == 200
    data = response.json()
    assert "psi" in data
    assert "threshold" in data
    assert "drift_detected" in data
    assert "status" in data
    assert data["threshold"] == 0.2


def test_drift_report_psi_is_number():
    """Test PSI is a valid number."""
    response = client.get("/drift-report")
    data = response.json()
    assert isinstance(data["psi"], (int, float))
    assert data["psi"] >= 0


# ============================================
# TEST 4: Explain endpoint
# ============================================
def test_explain_returns_shap():
    """Test SHAP explanation returns feature contributions."""
    response = client.get("/explain", params={
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    })
    assert response.status_code == 200
    data = response.json()
    assert "shap_values" in data
    assert "feature_importance" in data
    assert "flower" in data


def test_explain_has_all_features():
    """Test SHAP returns all 4 features."""
    response = client.get("/explain", params={
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    })
    data = response.json()
    shap_values = data["shap_values"]
    assert "sepal_length" in shap_values
    assert "sepal_width" in shap_values
    assert "petal_length" in shap_values
    assert "petal_width" in shap_values


# ============================================
# TEST 5: Add data endpoint
# ============================================
def test_add_data_returns_count():
    """Test that adding data returns updated count."""
    response = client.post("/add-data", params={
        "sepal_length": 8.0,
        "sepal_width": 5.0,
        "petal_length": 7.0,
        "petal_width": 4.0
    })
    assert response.status_code == 200
    data = response.json()
    assert "total_records" in data
    assert "current_psi" in data
    assert data["total_records"] > 0


# ============================================
# TEST 6: Auto-retrain endpoint
# ============================================
def test_auto_retrain_responds():
    """Test auto-retrain endpoint responds with expected structure."""
    response = client.post("/auto-retrain")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "drift_detected" in data
    assert "action" in data