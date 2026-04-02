# MIT License
#
# Copyright (c) 2024 Durai Rajamanickam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ============================================================
# Simulating causal data with treatment effect
# ============================================================

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


# ============================================================
# Chapter 10: Focusing on simulating causal data and implementing the evaluation metrics (PEHE and ATE Error)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error  # For a different perspective


# --- 1. Simulate Causal Data ---
def simulate_causal_data(n=1000, input_dim=5, ate=2.0):
    """Simulates causal data with confounding."""

    X = np.random.normal(0, 1, size=(n, input_dim)).astype(np.float32)  # Covariates
    true_treatment_effect = ate * np.ones(n)  # Constant treatment effect for simplicity

    # Simulate treatment assignment (propensity scores)
    propensity_logits = X[:, 0] - 0.5 * X[:, 1]  # Example: depends on first two covariates
    propensity_scores = 1 / (1 + np.exp(-propensity_logits))
    T = np.random.binomial(1, propensity_scores, size=n).astype(np.float32)  # Treatment

    Y0 = np.dot(X, np.array([0.5, -0.3, 0.2, 0.1, -0.2]).reshape(input_dim, 1)).flatten() + np.random.normal(0, 1, n)
    Y1 = Y0 + true_treatment_effect + np.random.normal(0, 0.5, n)  # Heterogeneous effects
    Y = T * Y1 + (1 - T) * Y0  # Observed outcome

    return pd.DataFrame({
        'X0': X[:, 0], 'X1': X[:, 1], 'X2': X[:, 2], 'X3': X[:, 3], 'X4': X[:, 4],
        'T': T,
        'Y': Y,
        'Y0': Y0,
        'Y1': Y1,
        'true_effect': Y1 - Y0
    })


# --- 2. Implement PEHE ---
def calculate_pehe(y1_true, y0_true, y1_pred, y0_pred):
    """Calculates the Precision in Estimation of Heterogeneous Effects (PEHE)."""

    ite_true = y1_true - y0_true
    ite_pred = y1_pred - y0_pred
    return np.sqrt(np.mean((ite_true - ite_pred) ** 2))


# --- 3. Implement ATE Error ---
def calculate_ate_error(true_ate, predicted_ate):
    """Calculates the error in estimating the Average Treatment Effect (ATE)."""

    return np.abs(true_ate - predicted_ate)


# --- 4. Policy Risk (Simplified - Example) ---
def calculate_policy_risk(y1_pred, y0_pred, t_true, y_true, benefit_threshold=1.0):
    """Calculates a simplified policy risk.

    This is a basic example. Real policy risk calculation is often more complex.
    """

    # A simple policy: Treat if predicted benefit > threshold
    treat_if = (y1_pred - y0_pred) > benefit_threshold
    optimal_treat = (y_true > np.mean(y_true))  # Example: treat "high" outcomes

    # Calculate "loss" - difference in treatment decisions
    policy_error = np.mean(treat_if != optimal_treat)
    return policy_error


# --- 5. Main Execution ---
if __name__ == '__main__':
    np.random.seed(123)
    data = simulate_causal_data()

    # 5.1. Example: Splitting and "Predicting" (using true outcomes for illustration)
    X = data[['X0', 'X1', 'X2', 'X3', 'X4']].values
    Y = data['Y'].values
    T = data['T'].values
    Y0_true = data['Y0'].values
    Y1_true = data['Y1'].values
    true_effect = data['true_effect'].values

    X_train, X_test, Y_train, Y_test, T_train, T_test, Y0_train, Y0_test, Y1_train, Y1_test, true_effect_train, true_effect_test = train_test_split(
        X, Y, T, Y0_true, Y1_true, true_effect, test_size=0.2, random_state=42
    )

    # In a real scenario, you'd replace this with your model's predictions
    y0_pred = Y0_test + np.random.normal(0, 0.5, len(Y0_test))  # Add some noise to simulate predictions
    y1_pred = Y1_test + np.random.normal(0, 0.5, len(Y1_test))

 # 5.2. Calculate and Print Metrics
# Pass the true potential outcomes from the test set to calculate_pehe
pehe = calculate_pehe(Y1_test, Y0_test, y1_pred, y0_pred)
ate_true = np.mean(true_effect_test) # Also calculate ATE on the test set for consistency
ate_pred = np.mean(y1_pred - y0_pred)
ate_error = calculate_ate_error(ate_true, ate_pred)
policy_risk = calculate_policy_risk(y1_pred, y0_pred, T_test, Y_test)

print("--- Evaluation Metrics ---")
print(f"PEHE: {pehe:.4f}")
print(f"ATE Error: {ate_error:.4f}")
print(f"Policy Risk: {policy_risk:.4f}")

# 5.3. Visualization (Example)
plt.figure(figsize=(8, 6))
# Visualize true potential outcomes from the test set
plt.scatter(Y0_test, Y1_test, alpha=0.5)
plt.xlabel("Y0 (Outcome if Untreated)")
plt.ylabel("Y1 (Outcome if Treated)")
plt.title("True Potential Outcomes (Test Set)") # Update title for clarity
plt.plot([min(Y0_test), max(Y0_test)], [min(Y0_test), max(Y0_test)], color='red', linestyle='--')  # Line of equality
plt.show()

