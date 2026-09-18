"""
Benchmark script for Self-Healing ML Pipeline.
Measures F1, PSI, and latency.
"""
import time
import numpy as np
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import requests

# ============================================
# 1. BASELINE F1
# ============================================
print("="*50)
print("BENCHMARK RESULTS")
print("="*50)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_pred = model.predict(X_test)
baseline_f1 = f1_score(y_test, y_pred, average="weighted")

print(f"\n1. Baseline F1 Score: {baseline_f1:.4f}")

# ============================================
# 2. API LATENCY
# ============================================
print("\n2. API Latency (100 requests)")
print("   Measuring...")

URL = "http://127.0.0.1:8000/predict"

# Warmup
requests.get(URL, params={"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2})

latencies = []
for _ in range(100):
    start = time.time()
    requests.get(URL, params={"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2})
    latencies.append((time.time() - start) * 1000)  # ms

latencies = np.array(latencies)
print(f"   p50 (median): {np.percentile(latencies, 50):.1f} ms")
print(f"   p95:          {np.percentile(latencies, 95):.1f} ms")
print(f"   p99:          {np.percentile(latencies, 99):.1f} ms")
print(f"   mean:         {latencies.mean():.1f} ms")

# ============================================
# 3. DRIFT DETECTION LATENCY
# ============================================
print("\n3. Drift Report Latency")
start = time.time()
requests.get("http://127.0.0.1:8000/drift-report")
drift_latency = (time.time() - start) * 1000
print(f"   PSI calculation: {drift_latency:.1f} ms")

# ============================================
# 4. PSI BEFORE DRIFT
# ============================================
print("\n4. PSI Values")
r = requests.get("http://127.0.0.1:8000/drift-report").json()
print(f"   PSI (no drift): {r['psi']:.4f}")
print(f"   Drift detected: {r['drift_detected']}")

# ============================================
# 5. SIMULATE DRIFT
# ============================================
print("\n5. Simulating drift (adding 20 extreme samples)...")
for i in range(20):
    requests.post("http://127.0.0.1:8000/add-data", params={
        "sepal_length": 9.0 + np.random.rand(),
        "sepal_width": 6.0 + np.random.rand(),
        "petal_length": 8.0 + np.random.rand(),
        "petal_width": 5.0 + np.random.rand()
    })

r = requests.get("http://127.0.0.1:8000/drift-report").json()
print(f"   PSI (after drift): {r['psi']:.4f}")
print(f"   Drift detected: {r['drift_detected']}")
print(f"   Total samples: {r['total_samples']}")

# ============================================
# 6. POST-RETRAIN F1 (baseline retrain on same data)
# ============================================
print("\n6. Post-retrain F1 (simulated on combined data)")
from sklearn.ensemble import RandomForestClassifier

# Combined data = original + 20 extreme samples
X_combined = np.vstack([X, np.random.rand(20, 4) * 3 + 5])
y_combined = np.concatenate([y, np.random.randint(0, 3, 20)])

X_tr, X_te, y_tr, y_te = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)
new_model = RandomForestClassifier(n_estimators=100, random_state=42)
new_model.fit(X_tr, y_tr)
y_pred_new = new_model.predict(X_te)
new_f1 = f1_score(y_te, y_pred_new, average="weighted")

print(f"   Post-retrain F1: {new_f1:.4f}")
print(f"   Delta: {new_f1 - baseline_f1:+.4f}")

print("\n" + "="*50)
print("BENCHMARK COMPLETE")
print("="*50)