"""
Champion/Challenger model retraining with proper evaluation.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# ============================================
# CONFIGURATION
# ============================================
ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "model.pkl"
DATA_PATH = ROOT / "current_data.csv"
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
MARGIN = 0.01  # challenger must beat champion by at least 1 point


def run_retrain():
    """
    Retrain a challenger model and compare against champion on same holdout.
    """
    # ============================================
    # VALIDATE INPUTS
    # ============================================
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No new data found at {DATA_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No champion model found at {MODEL_PATH}")

    # ============================================
    # LOAD CHAMPION MODEL
    # ============================================
    champion = joblib.load(MODEL_PATH)

    # ============================================
    # LOAD DATA
    # ============================================
    X_original, y_original = load_iris(return_X_y=True)
    new_data = pd.read_csv(DATA_PATH)

    if new_data.empty:
        raise ValueError(f"New data at {DATA_PATH} is empty")
    
    # Drop rows where target is missing (NaN) — these are unlabeled simulation data
    if 'target' in new_data.columns:
        labeled_data = new_data.dropna(subset=['target']).copy()
        labeled_data['target'] = labeled_data['target'].astype(int)
    else:
        labeled_data = new_data.copy()
        labeled_data['target'] = 0  # fallback if no target column

    if labeled_data.empty:
        raise ValueError("No labeled data found in current_data.csv. Cannot retrain.")

    X_new = labeled_data[FEATURES].values
    y_new = labeled_data["target"].values

    # ============================================
    # SPLIT NEW DATA INTO TRAIN/TEST FOR CHALLENGER
    # ============================================
    X_new_train, X_new_test, y_new_train, y_new_test = train_test_split(
        X_new, y_new, test_size=0.4, random_state=42, stratify=y_new
    )

    # ============================================
    # TRAIN CHALLENGER ON COMBINED DATA
    # ============================================
    X_combined = np.vstack([X_original, X_new_train])
    y_combined = np.concatenate([y_original, y_new_train])

    challenger = RandomForestClassifier(n_estimators=100, random_state=42)
    challenger.fit(X_combined, y_combined)

    # ============================================
    # EVALUATE BOTH ON SAME HOLDOUT
    # ============================================
    champion_predictions = champion.predict(X_new_test)
    challenger_predictions = challenger.predict(X_new_test)

    champion_f1 = f1_score(y_new_test, champion_predictions, average="weighted")
    challenger_f1 = f1_score(y_new_test, challenger_predictions, average="weighted")

    # ============================================
    # PROMOTION DECISION
    # ============================================
    promoted = challenger_f1 >= champion_f1 + MARGIN

    if promoted:
        temp_path = MODEL_PATH.with_suffix(".tmp")
        joblib.dump(challenger, temp_path)
        temp_path.replace(MODEL_PATH)

    return {
        "champion_f1": round(champion_f1, 4),
        "challenger_f1": round(challenger_f1, 4),
        "f1_delta": round(challenger_f1 - champion_f1, 4),
        "holdout_size": len(y_new_test),
        "promoted": promoted,
        "decision": "PROMOTED — challenger beat champion" if promoted else "KEPT — champion remained in production",
    }


if __name__ == "__main__":
    try:
        result = run_retrain()
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"RETRAIN FAILED: {e}", file=sys.stderr)
        sys.exit(1)