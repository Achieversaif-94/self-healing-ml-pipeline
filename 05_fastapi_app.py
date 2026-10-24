from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import shap
import pickle
import joblib
import os
import time
from datetime import datetime
from pathlib import Path

# ============================================
# IMPORT RETRAIN FUNCTION
# ============================================
from retrain import run_retrain, MODEL_PATH

# Initialize FastAPI
app = FastAPI(title="Iris Classifier API with Self-Healing")

# ============================================
# CORS MIDDLEWARE
# ============================================
allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# LOAD MODEL (supports both pickle and joblib)
# ============================================
print("Loading model...")
try:
    model = joblib.load(MODEL_PATH)
except Exception:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
print("Model loaded!")

# ============================================
# CREATE SHAP EXPLAINER
# ============================================
print("Creating SHAP explainer...")
explainer = shap.TreeExplainer(model)
print("SHAP explainer ready!")

iris_names = ['setosa', 'versicolor', 'virginica']

# ============================================
# TRAINING DATA (Reference)
# ============================================
X_train, y_train = load_iris(return_X_y=True)
training_data = pd.DataFrame(X_train, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

# ============================================
# CURRENT DATA (Starts as copy of training data)
# ============================================
current_data = training_data.copy()

# ============================================
# DRIFT HISTORY (for charting)
# ============================================
drift_history = []

# ============================================
# COOLDOWN TRACKER
# ============================================
last_retrain_time = 0

# ============================================
# REAL PSI CALCULATION
# ============================================
def calculate_psi(expected, actual, bins=10, epsilon=1e-6):
    """
    Calculate Population Stability Index (PSI) between two distributions.
    
    PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
    """
    expected = np.asarray(expected).flatten()
    actual = np.asarray(actual).flatten()
    
    breakpoints = np.linspace(
        min(expected.min(), actual.min()),
        max(expected.max(), actual.max()),
        bins + 1
    )
    
    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)
    
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)
    
    expected_pct = np.where(expected_pct == 0, epsilon, expected_pct)
    actual_pct = np.where(actual_pct == 0, epsilon, actual_pct)
    
    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    return float(np.sum(psi_values))


def calculate_drift():
    """Calculate drift across all 4 features using real PSI."""
    try:
        feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        
        psi_scores = []
        for col in feature_cols:
            psi = calculate_psi(
                training_data[col].values,
                current_data[col].values
            )
            psi_scores.append(psi)
        
        psi_avg = float(np.mean(psi_scores))
        drift_detected = any(p > 0.2 for p in psi_scores)
        
        return psi_avg, drift_detected
    except Exception as e:
        print(f"Drift calculation error: {e}")
        return 0.0, False


# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
def root():
    return {
        "message": "Iris Classifier API is running with Self-Healing!",
        "endpoints": ["/predict", "/drift-report", "/explain", "/add-data", "/auto-retrain"]
    }


@app.get("/predict")
def predict(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    pred = model.predict(features)[0]
    
    return {
        "prediction": int(pred),
        "flower": iris_names[pred],
        "features": {
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width
        }
    }


@app.get("/drift-report")
def drift_report():
    psi, detected = calculate_drift()
    
    drift_history.append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "psi": round(psi, 4)
    })
    if len(drift_history) > 20:
        drift_history.pop(0)
    
    return {
        "psi": psi,
        "threshold": 0.2,
        "drift_detected": detected,
        "status": "⚠️ Drift detected!" if detected else "✅ No drift",
        "total_samples": len(current_data),
        "history": drift_history
    }


@app.get("/explain")
def explain(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
    try:
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        shap_values = explainer.shap_values(features)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        pred = model.predict(features)[0]
        
        if isinstance(shap_values, np.ndarray):
            shap_list = shap_values.flatten().tolist()
        else:
            shap_list = [float(shap_values[i]) for i in range(4)]
        
        return {
            "prediction": int(pred),
            "flower": iris_names[pred],
            "shap_values": {
                feature_names[i]: round(shap_list[i], 4) for i in range(4)
            },
            "feature_importance": {
                feature_names[i]: round(abs(shap_list[i]), 4) for i in range(4)
            }
        }
    except Exception as e:
        return {"error": str(e), "message": "SHAP calculation error"}


@app.post("/add-data")
def add_data(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
    global current_data
    
    new_row = pd.DataFrame({
        'sepal_length': [sepal_length],
        'sepal_width': [sepal_width],
        'petal_length': [petal_length],
        'petal_width': [petal_width]
    })
    
    current_data = pd.concat([current_data, new_row], ignore_index=True)
    
    psi, detected = calculate_drift()
    
    return {
        "message": "Data added successfully!",
        "total_records": len(current_data),
        "drift_status": "⚠️ Drift detected!" if detected else "✅ No drift yet",
        "current_psi": psi
    }


@app.post("/auto-retrain")
def auto_retrain():
    """
    Trigger retraining if drift is detected.
    Uses run_retrain() function directly (no subprocess).
    Reloads model after promotion.
    """
    global model, explainer, last_retrain_time, current_data
    
    psi, detected = calculate_drift()
    
    if not detected:
        return {
            "action": "none",
            "psi": psi,
            "drift_detected": False,
            "message": "No drift detected. Retraining not needed."
        }
    
    # Cooldown: 60 seconds between retrains
    if time.time() - last_retrain_time < 60:
        raise HTTPException(
            status_code=429,
            detail=f"Cooldown active. Wait {int(60 - (time.time() - last_retrain_time))} seconds."
        )
    
    last_retrain_time = time.time()
    
    # Run retraining (raises exception on failure → 500 error, no silent failures)
    result = run_retrain()
    
    # Reload model if promoted
    if result["promoted"]:
        model = joblib.load(MODEL_PATH)
        explainer = shap.TreeExplainer(model)
    
    # ALWAYS reset drift reference after retraining (promoted or not)
    # This prevents infinite retraining loops when PSI stays high
    global training_data
    X_combined = np.vstack([X_train, current_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values])
    training_data = pd.DataFrame(X_combined, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    current_data = training_data.copy()

    return {
        "action": "retrained",
        "psi": psi,
        "drift_detected": True,
        **result
    }