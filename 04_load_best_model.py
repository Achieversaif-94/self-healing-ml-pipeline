import mlflow
from sklearn.datasets import load_iris

# REPLACE WITH YOUR RANDOMFOREST RUN ID
RUN_ID = "848fe30ea59047b0910d994112b866dc"

print(f"Loading RandomForest model from Run ID: {RUN_ID}")

# Load the model
model_path = f"runs:/{RUN_ID}/RandomForest_model"
model = mlflow.sklearn.load_model(model_path)

print("Model loaded successfully!")

# Test it
X, y = load_iris(return_X_y=True)
sample = X[:5]
predictions = model.predict(sample)

iris_names = ['setosa', 'versicolor', 'virginica']
print("\nTest predictions:")
for i, pred in enumerate(predictions):
    print(f"  Sample {i+1}: {iris_names[pred]}")

print("\nModel is ready for deployment!")