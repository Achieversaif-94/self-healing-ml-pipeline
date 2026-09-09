import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("="*50)
print("TRAINING 2 MODELS WITH MLFLOW")
print("="*50)

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models - ONLY 2, NO XGBOOST
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100)
}

# Train and log each model
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    with mlflow.start_run(run_name=name):
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log everything
        mlflow.log_param("model_name", name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log model
        mlflow.sklearn.log_model(model, f"{name}_model")
        
        run_id = mlflow.active_run().info.run_id
        print(f"Done! F1: {f1:.3f}, Accuracy: {accuracy:.3f}")
        print(f"Run ID: {run_id}")

print("\n" + "="*50)
print("ALL MODELS TRAINED!")
print("Run: mlflow ui")
print("="*50)