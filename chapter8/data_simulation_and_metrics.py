import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Simulate covariates
n = 1000
X = np.random.normal(0, 1, size=(n, 5))

# Simulate propensity scores and treatment assignment
ps_model = LogisticRegression()
ps_model.fit(X, np.random.binomial(1, 0.5, size=n))
propensity_scores = ps_model.predict_proba(X)[:, 1]
T = np.random.binomial(1, propensity_scores)

# Simulate potential outcomes
Y0 = np.dot(X, np.array([0.5, -0.3, 0.2, 0.1, -0.2])) + np.random.normal(0, 1, n)
Y1 = Y0 + 2.0  # Constant treatment effect

# Observed outcome
Y = T * Y1 + (1 - T) * Y0

# Create DataFrame
data = pd.DataFrame(np.column_stack((X, T, Y, Y0, Y1)), columns=['X1', 'X2', 'X3', 'X4', 'X5', 'T', 'Y', 'Y0', 'Y1'])


#Show propensity score distribution
plt.hist(propensity_scores, bins=20)
plt.title("Propensity Score Distribution")
plt.xlabel("Propensity Score")
plt.ylabel("Frequency")
plt.show()


# Example of calculating PEHE
def pehe(y1_true, y0_true, y1_pred, y0_pred):
    return np.mean((y1_true - y0_true - (y1_pred - y0_pred))**2)

y1_pred = Y0 + 2.1 + np.random.normal(0, 0.1, n)  # Slightly noisy prediction
y0_pred = Y0 + np.random.normal(0, 0.1, n)
pehe_value = pehe(Y1, Y0, y1_pred, y0_pred)
print(f"PEHE: {pehe_value:.4f}")

# Example of calculating ATE error
def ate_error(y1_true, y0_true, y1_pred, y0_pred):
    return np.abs(np.mean(y1_pred - y0_pred) - np.mean(y1_true - y0_true))

ate_error_value = ate_error(Y1, Y0, y1_pred, y0_pred)
print(f"ATE Error: {ate_error_value:.4f}")