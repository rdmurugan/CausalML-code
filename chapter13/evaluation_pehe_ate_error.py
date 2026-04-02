import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # Example model
from sklearn.ensemble import RandomForestRegressor  # Another example
from sklearn.metrics import mean_squared_error

# --- 1. Simulate Causal Data ---
def simulate_causal_data(n=1000, input_dim=5, ate_mean=2.0, ate_std=1.0):
    """Simulates causal data with heterogeneous treatment effects and confounding.

    Args:
        n: Number of samples.
        input_dim: Number of input features.
        ate_mean: Mean of the Average Treatment Effect.
        ate_std: Standard deviation of the treatment effect heterogeneity.

    Returns:
        Pandas DataFrame with simulated data.
    """

    X = np.random.normal(0, 1, size=(n, input_dim)).astype(np.float32)  # Covariates
    true_individual_effect = np.random.normal(ate_mean, ate_std, size=n).astype(np.float32)
    true_ate = np.mean(true_individual_effect)

    # Propensity scores (probability of treatment)
    propensity_logits = X[:, 0] - 0.5 * X[:, 1] + 0.2 * X[:, 2]  # Example: depends on first 3 covariates
    propensity_scores = 1 / (1 + np.exp(-propensity_logits))
    T = np.random.binomial(1, propensity_scores, size=n).astype(np.float32)  # Treatment

    Y0 = np.dot(X, np.array([0.5, -0.3, 0.2, 0.1, -0.2]).reshape(input_dim, 1)).flatten() + np.random.normal(0, 1, n)
    Y1 = Y0 + true_individual_effect + np.random.normal(0, 0.5, n)  # Heterogeneous effects
    Y = T * Y1 + (1 - T) * Y0  # Observed outcome

    return pd.DataFrame({
        'X0': X[:, 0], 'X1': X[:, 1], 'X2': X[:, 2], 'X3': X[:, 3], 'X4': X[:, 4],
        'T': T,
        'Y': Y,
        'Y0': Y0,
        'Y1': Y1,
        'true_effect': true_individual_effect
    }), true_ate


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


# --- 4. Model Training and Prediction (Example) ---
def train_and_predict(model, X_train, Y_train, X_test):
    """Trains a model and makes predictions. (Illustrative)"""

    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    return Y_pred


# --- 5. Main Execution ---
if __name__ == '__main__':
    np.random.seed(42)

    # 5.1. Generate Data
    data, true_ate = simulate_causal_data()

    # 5.2. Split Data
    X = data[['X0', 'X1', 'X2', 'X3', 'X4']].values
    Y = data['Y'].values
    T = data['T'].values
    Y0_true = data['Y0'].values
    Y1_true = data['Y1'].values
    true_effect = data['true_effect'].values

    X_train, X_test, Y_train, Y_test, T_train, T_test, Y0_train, Y0_test, Y1_train, Y1_test, true_effect_train, true_effect_test = train_test_split(
        X, Y, T, Y0_true, Y1_true, true_effect, test_size=0.2, random_state=42
    )

# 5.3. Example: Linear Regression Model
model_linear = LinearRegression()
Y_pred_linear = train_and_predict(model_linear, X_train, Y_train, X_test)
# It's more appropriate to train separate models for Y0 and Y1 on the training data,
# using only the relevant subsets (T_train == 0 for Y0, T_train == 1 for Y1) if
# you were trying to replicate potential outcome estimation methods.
# However, for this illustrative example just predicting Y0_train and Y1_train
# directly on X_test as if they were observed is acceptable to demonstrate the metric calculation.
Y0_pred_linear = train_and_predict(model_linear, X_train, Y0_train, X_test)
Y1_pred_linear = train_and_predict(model_linear, X_train, Y1_train, X_test)

# Pass the true potential outcomes from the test set to calculate_pehe
pehe_linear = calculate_pehe(Y1_test, Y0_test, Y1_pred_linear, Y0_pred_linear)
ate_pred_linear = np.mean(Y1_pred_linear - Y0_pred_linear)
# Calculate the true ATE on the test set for direct comparison
ate_true_test = np.mean(true_effect_test)
ate_error_linear = calculate_ate_error(ate_true_test, ate_pred_linear) # Use test set ATE
mse_linear = mean_squared_error(Y_test, Y_pred_linear)

print("\n--- Linear Regression Evaluation ---")
print(f"PEHE: {pehe_linear:.4f}")
print(f"ATE Error: {ate_error_linear:.4f}")
print(f"MSE: {mse_linear:.4f}")

# 5.4. Example: Random Forest Model
model_rf = RandomForestRegressor(random_state=42)
Y_pred_rf = train_and_predict(model_rf, X_train, Y_train, X_test)
# Similar to the linear model, training on Y0_train and Y1_train directly
Y0_pred_rf = train_and_predict(model_rf, X_train, Y0_train, X_test) # Changed from Y0_true
Y1_pred_rf = train_and_predict(model_rf, X_train, Y1_train, X_test) # Changed from Y1_true

# Pass the true potential outcomes from the test set
pehe_rf = calculate_pehe(Y1_test, Y0_test, Y1_pred_rf, Y0_pred_rf) # Changed from Y1_true, Y0_true
ate_pred_rf = np.mean(Y1_pred_rf - Y0_pred_rf)
ate_error_rf = calculate_ate_error(ate_true_test, ate_pred_rf) # Use test set ATE
mse_rf = mean_squared_error(Y_test, Y_pred_rf)

print("\n--- Random Forest Evaluation ---")
print(f"PEHE: {pehe_rf:.4f}")
print(f"ATE Error: {ate_error_rf:.4f}")
print(f"MSE: {mse_rf:.4f}")

# 5.5. Visualization (Example)
plt.figure(figsize=(8, 6))
# Visualize true potential outcomes from the test set
plt.scatter(Y0_test, Y1_test, alpha=0.5, label="True Potential Outcomes (Test)") # Changed label
plt.scatter(Y0_pred_linear, Y1_pred_linear, alpha=0.5, label="Linear Pred")
plt.scatter(Y0_pred_rf, Y1_pred_rf, alpha=0.5, label="RF Pred")
plt.xlabel("Y0 (Outcome if Untreated)")
plt.ylabel("Y1 (Outcome if Treated)")
plt.title("Potential Outcomes: True (Test Set) vs. Predicted") # Changed title
plt.plot([min(Y0_test), max(Y0_test)], [min(Y0_test), max(Y0_test)], color='red', linestyle='--', label="Line of Equality") # Use test set for limits
plt.legend()
plt.show()

