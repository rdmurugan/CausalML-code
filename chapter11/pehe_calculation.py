import numpy as np
import pandas as pd

# Simulate data
np.random.seed(6)
n = 1000
X = np.random.normal(0, 1, (n, 2))
T = np.random.binomial(1, 1 / (1 + np.exp(-X[:, 0])), n) # Treatment depends on X
Y0 = X[:, 0] + np.random.normal(0, 1, n)
Y1 = Y0 + 5  # True ATE = 5
Y = T * Y1 + (1 - T) * Y0
data = pd.DataFrame({'X1': X[:, 0], 'X2': X[:, 1], 'T': T, 'Y': Y, 'Y0': Y0, 'Y1': Y1})


def pehe(y1_true, y0_true, y1_pred, y0_pred):
    ite_true = y1_true - y0_true
    ite_pred = y1_pred - y0_pred
    return np.sqrt(np.mean((ite_true - ite_pred) ** 2))

# Example (using data from Chapter 8 simulation)
y1_true = data['Y1'].values
y0_true = data['Y0'].values
y1_pred = y1_true + np.random.normal(0, 1, n)  # Simulated predictions
y0_pred = y0_true + np.random.normal(0, 1, n)
pehe_value = pehe(y1_true, y0_true, y1_pred, y0_pred)
print(f"PEHE: {pehe_value:.4f}")