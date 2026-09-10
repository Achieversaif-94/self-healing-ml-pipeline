import pandas as pd
import numpy as np
import os

print("Current directory:", os.getcwd())

# Create 30 extreme samples (drifted data)
np.random.seed(42)
n_samples = 30

new_data = pd.DataFrame({
    'sepal_length': np.random.uniform(7.5, 9.5, n_samples),
    'sepal_width': np.random.uniform(4.5, 6.0, n_samples),
    'petal_length': np.random.uniform(6.5, 8.5, n_samples),
    'petal_width': np.random.uniform(3.0, 5.0, n_samples),
    'target': np.random.randint(0, 3, n_samples)
})

# Save with full path
output_path = r"D:\P2\current_data.csv"
new_data.to_csv(output_path, index=False)

print(f"✅ Created {n_samples} new samples at {output_path}")
print(new_data.head())