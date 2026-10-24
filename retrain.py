"""
Champion/Challenger model retraining with proper evaluation.

Key fixes:
- Same holdout set for both models (fair F1 comparison)
- Actually updates model.pkl when challenger wins (real self-healing)
- Fails loudly on errors (no silent failures)
- Relative paths (works in Docker)
- Requires margin to prevent noise-driven promotions
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
    
    Returns dict with:
        champion_f1: F1 of current production model on holdout
        challenger_f1: F1 of newly trained model on holdout
        holdout_size: number of samples in holdout
        promoted: True if challenger was promoted to production
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

    X_new = new_data[FEATURES].values
    y_new = new_data["target"].values

    # ============================================
    # SPLIT NEW DATA INTO TRAIN/TEST FOR CHALLENGER
    # This is the SAME holdout used to evaluate BOTH models
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
        # Atomic swap: save to temp, then replace
        temp_path = MODEL_PATH.with_suffix(".tmp")
        joblib.dump(challenger, temp_path)
        temp_path.replace(MODEL_PATH)

    return {
        "champion_f1": round(champion_f1, 4),
        "challenger_f1": round(challenger_f1, 4),
        "holdout_size": len(y_new_test),
        "promoted": promoted,
    }


if __name__ == "__main__":
    try:
        result = run_retrain()
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"RETRAIN FAILED: {e}", file=sys.stderr)
        sys.exit(1)