"""
Auto-Retrain Script for Self-Healing ML Pipeline
Runs when drift is detected (PSI > 0.2)
"""

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import os

# ============================================
# CONFIGURATION
# ============================================
CURRENT_RUN_ID = "b78e8495603d45c9aa17a5e274455f09"  # Your current champion model
NEW_DATA_PATH = r"D:\P2\current_data.csv"  # Where we store new data
DRIFT_THRESHOLD = 0.2

print("=" * 50)
print("AUTO-RETRAIN SCRIPT")
print("=" * 50)

# ============================================
# STEP 1: Load Original Training Data
# ============================================
print("\n[1/5] Loading original training data...")
X_train, y_train = load_iris(return_X_y=True)

# ============================================
# STEP 2: Load New Data (If Exists)
# ============================================
print("[2/5] Checking for new data...")
if os.path.exists(NEW_DATA_PATH):
    new_data = pd.read_csv(NEW_DATA_PATH)
    print(f"   Found {len(new_data)} new samples")
    
    # Extract features and labels
    X_new = new_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
    y_new = new_data['target'].values
    
    # Combine old + new data
    X_combined = np.vstack([X_train, X_new])
    y_combined = np.concatenate([y_train, y_new])
    print(f"   Combined data: {len(X_combined)} samples")
else:
    X_combined = X_train
    y_combined = y_train
    print("   No new data found. Using original data only.")

# ============================================
# STEP 3: Split Data
# ============================================
print("[3/5] Splitting data...")
X_tr, X_te, y_tr, y_te = train_test_split(
    X_combined, y_combined, test_size=0.2, random_state=42
)

# ============================================
# STEP 4: Train New Model
# ============================================
print("[4/5] Training new RandomForest model...")

with mlflow.start_run(run_name="AutoRetrain_RandomForest"):
    # Train
    new_model = RandomForestClassifier(n_estimators=100, random_state=42)
    new_model.fit(X_tr, y_tr)
    
    # Evaluate
    y_pred = new_model.predict(X_te)
    new_f1 = f1_score(y_te, y_pred, average='weighted')
    
    # Log parameters
    mlflow.log_param("model_name", "RandomForest_AutoRetrain")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("trigger", "drift_detected")
    
    # Log metrics
    mlflow.log_metric("f1_score", new_f1)
    
    # Log model
    mlflow.sklearn.log_model(new_model, "RandomForest_AutoRetrain_model")
    
    new_run_id = mlflow.active_run().info.run_id
    
    print(f"   New model F1: {new_f1:.4f}")
    print(f"   New Run ID: {new_run_id}")

# ============================================
# STEP 5: Compare with Current Champion
# ============================================
print("[5/5] Comparing with current champion...")

try:
    client = mlflow.tracking.MlflowClient()
    current_run = client.get_run(CURRENT_RUN_ID)
    current_f1 = current_run.data.metrics.get("f1_score", 0)
    
    print(f"   Current champion F1: {current_f1:.4f}")
    print(f"   New model F1:        {new_f1:.4f}")
    
    if new_f1 > current_f1:
        print("\n   [OK] NEW MODEL IS BETTER!")
        print("   [OK] PROMOTING NEW MODEL TO PRODUCTION")
        print(f"   [OK] New production model: {new_run_id}")
        
        # Save promotion decision
        with open("model_promotion.txt", "w") as f:
            f.write(f"promoted_run_id={new_run_id}\n")
            f.write(f"f1_score={new_f1}\n")
            f.write(f"previous_f1={current_f1}\n")
    else:
        print("\n   [X] NEW MODEL IS WORSE")
        print("   [X] KEEPING OLD MODEL IN PRODUCTION")
except Exception as e:
    print(f"   Error comparing: {e}")
    print("   Keeping old model.")

print("\n" + "=" * 50)
print("AUTO-RETRAIN COMPLETE")
print("=" * 50)