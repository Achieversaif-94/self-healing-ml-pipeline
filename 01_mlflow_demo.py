import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("Starting MLflow demo...")

# 1. Load data
print("Loading Iris dataset...")
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train model
print("Training Random Forest...")
model = RandomForestClassifier(n_estimators=100, max_depth=3)
model.fit(X_train, y_train)

# 3. Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# 4. Log to MLflow
print("Logging to MLflow...")
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 3)
    mlflow.log_param("model_type", "RandomForest")

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)

    # Log model
    mlflow.sklearn.log_model(model, "iris_model")

    # Log the run ID
    run_id = mlflow.active_run().info.run_id
    print(f"Run ID: {run_id}")

print("Done! Now run: mlflow ui")