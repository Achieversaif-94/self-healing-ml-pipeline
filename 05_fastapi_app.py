from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import mlflow
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import shap

# Initialize FastAPI
app = FastAPI(title="Iris Classifier API with SHAP")

# ============================================
# CORS MIDDLEWARE
# ============================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# YOUR RANDOMFOREST RUN ID
# ============================================
RUN_ID = "b78e8495603d45c9aa17a5e274455f09"

# ============================================
# LOAD MODEL
# ============================================
print("Loading model...")
model = mlflow.sklearn.load_model(f"runs:/{RUN_ID}/RandomForest_model")
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
# DRIFT DETECTION
# ============================================
def calculate_drift():
    try:
        train_numeric = training_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
        current_numeric = current_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
        
        train_mean = train_numeric.mean()
        current_mean = current_numeric.mean()
        
        denominator = train_mean.abs() + 0.001
        diff = abs((current_mean - train_mean) / denominator).mean()
        
        psi = min(1.0, float(diff * 2))
        return psi, psi > 0.2
    except Exception as e:
        print(f"Drift calculation error: {e}")
        return 0.0, False

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
def root():
    return {
        "message": "Iris Classifier API is running with SHAP!",
        "endpoints": ["/predict", "/drift-report", "/explain", "/add-data"]
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
    return {
        "psi": psi,
        "threshold": 0.2,
        "drift_detected": detected,
        "status": "⚠️ Drift detected!" if detected else "✅ No drift",
        "total_samples": len(current_data)
    }

@app.get("/explain")
def explain(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
    try:
        # Convert input to array
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(features)
        
        # Handle different shap output formats
        if isinstance(shap_values, list):
            # For classification, take first class
            shap_values = shap_values[0]
        
        # Get feature names
        feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        
        # Get prediction
        pred = model.predict(features)[0]
        
        # Convert shap values to list
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
            },
            "explanation": "Higher positive values push prediction toward this flower"
        }
    except Exception as e:
        return {
            "error": str(e),
            "message": "SHAP calculation error. Using fallback.",
            "features": {
                "sepal_length": sepal_length,
                "sepal_width": sepal_width,
                "petal_length": petal_length,
                "petal_width": petal_width
            }
        }

@app.post("/add-data")
def add_data(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
    global current_data
    
    print(f"📥 Adding data: sl={sepal_length}, sw={sepal_width}, pl={petal_length}, pw={petal_width}")
    
    new_row = pd.DataFrame({
        'sepal_length': [sepal_length],
        'sepal_width': [sepal_width],
        'petal_length': [petal_length],
        'petal_width': [petal_width]
    })
    
    current_data = pd.concat([current_data, new_row], ignore_index=True)
    
    print(f"📊 Total records: {len(current_data)}")
    
    psi, detected = calculate_drift()
    
    return {
        "message": "Data added successfully!",
        "total_records": len(current_data),
        "drift_status": "⚠️ Drift detected!" if detected else "✅ No drift yet",
        "current_psi": psi
    }