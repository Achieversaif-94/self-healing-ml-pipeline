"""
Generate shifted data to simulate real drift.

Instead of random noise, we take real Iris samples and shift their features.
This creates realistic drift that a model would genuinely struggle with,
making the PSI test and champion/challenger comparison meaningful.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

# ============================================
# CONFIGURATION
# ============================================
# How much to shift each feature (in standard deviations)
SHIFT_FACTORS = {
    'sepal_length': 1.5,
    'sepal_width': 1.0,
    'petal_length': 2.5,
    'petal_width': 1.5
}

N_SAMPLES = 50

# ============================================
# GENERATE SHIFTED DATA
# ============================================
def generate_shifted_data():
    print("Loading original Iris data...")
    X, y = load_iris(return_X_y=True)
    
    # Calculate standard deviations for scaling the shift
    stds = X.std(axis=0)
    means = X.mean(axis=0)
    
    # Take random real samples
    np.random.seed(42)  # Fixed seed for reproducibility
    indices = np.random.choice(len(X), N_SAMPLES, replace=False)
    X_sample = X[indices].copy()
    y_sample = y[indices].copy()
    
    # Apply the shift to simulate distribution change
    # We shift features in the direction that makes them look more like "virginica"
    # while keeping the labels the same (simulating label drift)
    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    
    print(f"Applying shift factors: {SHIFT_FACTORS}")
    
    for i, feature in enumerate(feature_names):
        shift_amount = SHIFT_FACTORS[feature] * stds[i]
        X_sample[:, i] += shift_amount
        print(f"   {feature}: shifted by +{shift_amount:.2f} (std: {stds[i]:.2f})")
    
    # Create DataFrame
    df = pd.DataFrame(X_sample, columns=feature_names)
    df['target'] = y_sample
    
    # Save
    output_path = "current_data.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Generated {N_SAMPLES} shifted samples")
    print(f"✅ Saved to {output_path}")
    
    # Show the drift magnitude
    original_mean = X.mean(axis=0)
    shifted_mean = X_sample.mean(axis=0)
    
    print("\n📊 Mean Shift (original → shifted):")
    for i, feature in enumerate(feature_names):
        print(f"   {feature}: {original_mean[i]:.2f} → {shifted_mean[i]:.2f}")
    
    return df

if __name__ == "__main__":
    generate_shifted_data()