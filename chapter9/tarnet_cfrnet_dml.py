"""
Chapter 9: Causal Inference Meets Deep Learning
From "Causal Inference for Machine Learning Engineers - A Practical Guide"
by Durai Rajamanickam
"""

# ============================================================
# Code Block 1: Simple Double Machine Learning Implementation
# ============================================================

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Simulate data
np.random.seed(42)
n = 1000
X = np.random.normal(0, 1, size=(n, 5))
T = np.random.binomial(1, 0.5, size=n)
Y = 2 * T + np.dot(X, [0.5, -0.3, 0.2, 0.1, -0.2]) + np.random.normal(0, 1, n)

# Step 1: Fit nuisance models
g_model = RandomForestRegressor().fit(X, Y)
e_model = RandomForestRegressor().fit(X, T)

# Step 2: Compute residuals
g_pred = g_model.predict(X)
e_pred = e_model.predict(X)

Y_res = Y - g_pred
T_res = T - e_pred

# Step 3: Regress residuals
reg = LinearRegression().fit(T_res.reshape(-1, 1), Y_res)
theta_hat = reg.coef_[0]

print(f"Estimated treatment effect (theta): {theta_hat:.4f}")


# ============================================================
# Code Block 2: Chapter 7: A basic TARNet-like structure
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- 1. Simulate Data ---
def simulate_treatment_effect_data(n=1000, input_dim=5, ate=2.0):
    """Simulates data with treatment effect, similar to IHDP.

    Args:
        n: Number of samples.
        input_dim: Number of input features.
        ate: Average Treatment Effect.

    Returns:
        X, T, Y, Y0, Y1 (torch tensors)
    """

    X = np.random.rand(n, input_dim).astype(np.float32)  # Covariates
    T = np.random.binomial(1, 0.5, n).astype(np.float32)  # Treatment (binary)
    Y0 = np.dot(X, np.random.rand(input_dim, 1)).flatten() + np.random.normal(0, 1, n)  # Outcome if T=0
    Y1 = Y0 + ate + np.random.normal(0, 1, n)  # Outcome if T=1
    Y = T * Y1 + (1 - T) * Y0  # Observed outcome

    # Explicitly specify dtype=torch.float32 for all tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    T_tensor = torch.tensor(T, dtype=torch.float32).reshape(-1, 1)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).reshape(-1, 1)
    Y0_tensor = torch.tensor(Y0, dtype=torch.float32).reshape(-1, 1)  # For evaluation
    Y1_tensor = torch.tensor(Y1, dtype=torch.float32).reshape(-1, 1)

    return X_tensor, T_tensor, Y_tensor, Y0_tensor, Y1_tensor

# --- 2. TARNet Architecture ---
class TARNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=100):
        super(TARNet, self).__init__()
        self.shared_representation = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.outcome_T0 = nn.Linear(hidden_dim, 1)  # Outcome head for T=0
        self.outcome_T1 = nn.Linear(hidden_dim, 1)  # Outcome head for T=1

    def forward(self, x, t):
        shared = self.shared_representation(x)
        y0_hat = self.outcome_T0(shared)
        y1_hat = self.outcome_T1(shared)
        y_hat = t * y1_hat + (1 - t) * y0_hat  # Observed outcome prediction
        return y_hat, y0_hat, y1_hat


# --- 3. CFRNet Architecture (Simplified Balancing) ---
class CFRNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=100, ipm_lambda=0.1):  # ipm_lambda: balancing strength
        super(CFRNet, self).__init__()
        self.shared_representation = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.outcome_T0 = nn.Linear(hidden_dim, 1)
        self.outcome_T1 = nn.Linear(hidden_dim, 1)
        self.ipm_lambda = ipm_lambda

    def forward(self, x, t):
        shared = self.shared_representation(x)
        y0_hat = self.outcome_T0(shared)
        y1_hat = self.outcome_T1(shared)
        y_hat = t * y1_hat + (1 - t) * y0_hat

        return y_hat, y0_hat, y1_hat, shared  # Return shared for balancing

    def calculate_balancing_loss(self, shared, t):
        """Simplified balancing loss: difference in means."""
        shared_t1 = shared[t.flatten() == 1]
        shared_t0 = shared[t.flatten() == 0]
        if shared_t1.shape[0] == 0 or shared_t0.shape[0] == 0:  # Handle edge case
            # Ensure the returned tensor also has the correct dtype
            return torch.tensor(0.0, requires_grad=True, dtype=torch.float32)
        # Ensure the result of the mean difference is also float32
        return (torch.mean(shared_t1, dim=0) - torch.mean(shared_t0, dim=0))

# --- 4. Training Function ---
def train_model(model, X_train, T_train, Y_train, optimizer, epochs=100, batch_size=32, model_type="TARNet"):
    """Trains TARNet or CFRNet."""

    dataset = TensorDataset(X_train, T_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    model.train()

    for epoch in range(epochs):
        for x_batch, t_batch, y_batch in dataloader:
            optimizer.zero_grad()
            if model_type == "CFRNet":
                y_pred, _, _, shared_batch = model(x_batch, t_batch)
                prediction_loss = criterion(y_pred, y_batch)
                # Ensure balancing_loss is computed and added as float32
                balancing_loss = model.calculate_balancing_loss(shared_batch, t_batch).mean()
                loss = prediction_loss + model.ipm_lambda * balancing_loss
            else:  # TARNet
                y_pred, _, _ = model(x_batch, t_batch)
                loss = criterion(y_pred, y_batch)

            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}, Loss: {loss.item()}')


# --- 5. Evaluation Function ---
def evaluate_model(model, X_test, T_test, Y_test, Y0_test, Y1_test):
    """Evaluates the trained model."""

    model.eval()
    with torch.no_grad():
        # Check model type to handle different return values
        if isinstance(model, CFRNet):
            y_pred, y0_pred, y1_pred, _ = model(X_test, T_test) # Unpack 4, ignore the 4th
        else: # Assume TARNet or similar returning 3 values
            y_pred, y0_pred, y1_pred = model(X_test, T_test)

        y_pred = y_pred.squeeze().numpy()
        y0_pred = y0_pred.squeeze().numpy()
        y1_pred = y1_pred.squeeze().numpy()
        Y_test = Y_test.squeeze().numpy()
        Y0_test = Y0_test.squeeze().numpy()
        Y1_test = Y1_test.squeeze().numpy()

        mse = mean_squared_error(Y_test, y_pred)
        ate_pred = np.mean(y1_pred - y0_pred)
        ate_true = np.mean(Y1_test - Y0_test)
        ate_error = np.abs(ate_pred - ate_true)

        print("\n--- Evaluation ---")
        print(f"MSE: {mse:.4f}")
        print(f"Predicted ATE: {ate_pred:.4f}")
        print(f"True ATE: {ate_true:.4f}")
        print(f"ATE Error: {ate_error:.4f}")

# --- 6. Main Execution ---
if __name__ == '__main__':
    np.random.seed(42)
    X, T, Y, Y0, Y1 = simulate_treatment_effect_data()
    X_train, X_test, T_train, T_test, Y_train, Y_test, Y0_train, Y0_test, Y1_train, Y1_test = train_test_split(
        X, T, Y, Y0, Y1, test_size=0.2
    )

    # 6.1. Train and Evaluate TARNet
    tarnet = TARNet(input_dim=X.shape[1])
    optimizer_tarnet = optim.Adam(tarnet.parameters(), lr=0.01)
    train_model(tarnet, X_train, T_train, Y_train, optimizer_tarnet, model_type="TARNet")
    evaluate_model(tarnet, X_test, T_test, Y_test, Y0_test, Y1_test)

    # 6.2. Train and Evaluate CFRNet
    cfrnet = CFRNet(input_dim=X.shape[1], ipm_lambda=0.1)  # Experiment with ipm_lambda
    optimizer_cfrnet = optim.Adam(cfrnet.parameters(), lr=0.01)
    train_model(cfrnet, X_train, T_train, Y_train, optimizer_cfrnet, model_type="CFRNet")
    # Correcting the order of arguments in evaluate_model for CFRNet
    evaluate_model(cfrnet, X_test, T_test, Y_test, Y0_test, Y1_test)

    print("\n--- Explanation ---")
    print("This code demonstrates simplified TARNet and CFRNet implementations.")
    print("TARNet learns a shared representation and has separate outcome heads.")
    print("CFRNet adds a balancing loss to TARNet to encourage similar representations for treated and control groups.")
    print("The simulation generates synthetic data with a known Average Treatment Effect (ATE).")
    print("Evaluation compares the models' ability to predict outcomes and estimate the ATE.")

