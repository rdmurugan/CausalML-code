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
# Simple CFRNet Implementation in PyTorch
# ============================================================

import torch
import torch.nn as nn

class CFRNet(nn.Module):
    def __init__(self, input_dim):
        super(CFRNet, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU()
        )
        self.head_treated = nn.Sequential(
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        self.head_control = nn.Sequential(
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        
    def forward(self, x, t):
        shared = self.shared(x)
        yt = self.head_treated(shared)
        yc = self.head_control(shared)
        return t * yt + (1 - t) * yc


# ============================================================
# Evaluating CFRNet with PEHE and ATE Error
# ============================================================

# Predict outcomes under treatment and control
with torch.no_grad():
    y0_pred = model(X, torch.zeros_like(T)).squeeze().numpy()
    y1_pred = model(X, torch.ones_like(T)).squeeze().numpy()

# Calculate true treatment effects
true_effect = Y1 - Y0
pred_effect = y1_pred - y0_pred

# PEHE
pehe = np.sqrt(np.mean((pred_effect - true_effect) ** 2))

# ATE Error
true_ate = np.mean(true_effect)
pred_ate = np.mean(pred_effect)
ate_error = np.abs(pred_ate - true_ate)

print(f"PEHE: {pehe:.4f}")
print(f"ATE Error: {ate_error:.4f}")


# ============================================================
# Chapter 11: Focusing on implementing the CFRNet architecture in PyTorch
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
    """Simulates data with treatment effect and confounding."""

    X = np.random.rand(n, input_dim).astype(np.float32)  # Covariates - Explicitly cast to float32
    true_treatment_effect = ate * np.ones(n, dtype=np.float32) # Explicitly cast to float32

    # Simulate treatment assignment with confounding
    propensity_logits = X[:, 0] - 0.5 * X[:, 1]  # Example: depends on first two covariates
    propensity_scores = 1 / (1 + np.exp(-propensity_logits))
    T = np.random.binomial(1, propensity_scores, size=n).astype(np.float32) # Treatment - Explicitly cast to float32

    Y0_np = np.dot(X, np.random.rand(input_dim, 1).astype(np.float32)).flatten() + np.random.normal(0, 1, n).astype(np.float32) # Explicitly cast to float32
    Y1_np = Y0_np + true_treatment_effect + np.random.normal(0, 0.5, n).astype(np.float32) # Explicitly cast to float32
    Y_np = T * Y1_np + (1 - T) * Y0_np  # Observed outcome

    # Convert to PyTorch tensors with explicit dtype
    X_tensor = torch.tensor(X, dtype=torch.float32)
    T_tensor = torch.tensor(T, dtype=torch.float32).reshape(-1, 1)
    Y_tensor = torch.tensor(Y_np, dtype=torch.float32).reshape(-1, 1)
    Y0_tensor = torch.tensor(Y0_np, dtype=torch.float32).reshape(-1, 1)
    Y1_tensor = torch.tensor(Y1_np, dtype=torch.float32).reshape(-1, 1)


    return X_tensor, T_tensor, Y_tensor, Y0_tensor, Y1_tensor


# --- 2. CFRNet Architecture (Simplified Balancing) ---
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
            # Ensure the returned tensor matches the expected dtype (float32)
            return torch.tensor(0.0, requires_grad=True, dtype=torch.float32)
        # Ensure the mean calculation and result are float32
        return torch.mean(shared_t1, dim=0).to(torch.float32) - torch.mean(shared_t0, dim=0).to(torch.float32)


# --- 3. Training Function ---
def train_cfrnet(model, X_train, T_train, Y_train, optimizer, epochs=100, batch_size=32):
    """Trains CFRNet."""

    dataset = TensorDataset(X_train, T_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    model.train()

    for epoch in range(epochs):
        for x_batch, t_batch, y_batch in dataloader:
            optimizer.zero_grad()
            # Ensure inputs to the model have the correct dtype if they weren't already
            x_batch = x_batch.to(torch.float32)
            t_batch = t_batch.to(torch.float32)
            y_batch = y_batch.to(torch.float32)

            y_pred, _, _, shared_batch = model(x_batch, t_batch)
            prediction_loss = criterion(y_pred, y_batch)
            # Ensure balancing_loss is calculated on float32 shared representations
            balancing_loss = model.calculate_balancing_loss(shared_batch.to(torch.float32), t_batch).mean()
            loss = prediction_loss + model.ipm_lambda * balancing_loss
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}, Loss: {loss.item()}')


# --- 4. Evaluation Function ---
def evaluate_cfrnet(model, X_test, T_test, Y_test, Y0_test, Y1_test):
    """Evaluates the trained CFRNet model."""

    model.eval()
    with torch.no_grad():
        # Ensure test data is also float32
        X_test = X_test.to(torch.float32)
        T_test = T_test.to(torch.float32)
        Y_test = Y_test.to(torch.float32)
        Y0_test = Y0_test.to(torch.float32)
        Y1_test = Y1_test.to(torch.float32)

        y_pred, y0_pred, y1_pred, _ = model(X_test, T_test)  # Get all outputs
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
        pehe = calculate_pehe(Y1_test, Y0_test, y1_pred, y0_pred)

        print("\n--- Evaluation ---")
        print(f"MSE: {mse:.4f}")
        print(f"Predicted ATE: {ate_pred:.4f}")
        print(f"True ATE: {ate_true:.4f}")
        print(f"ATE Error: {ate_error:.4f}")
        print(f"PEHE: {pehe:.4f}")


# The calculate_pehe function from the first cell is fine, but it's good practice
# to include it in the cell where it's used if you're running them independently.
# Since it was defined in the first cell and the second cell redefines it,
# the definition in the second cell is what will be used. Let's keep the one
# from the second cell.
def calculate_pehe(y1_true, y0_true, y1_pred, y0_pred):
    """Calculates the Precision in Estimation of Heterogeneous Effects (PEHE)."""
    ite_true = y1_true - y0_true
    ite_pred = y1_pred - y0_pred
    return np.sqrt(np.mean((ite_true - ite_pred) ** 2))


# --- 5. Main Execution ---
if __name__ == '__main__':
    np.random.seed(42)

    # 5.1. Generate Data
    X, T, Y, Y0, Y1 = simulate_treatment_effect_data()
    X_train, X_test, T_train, T_test, Y_train, Y_test, Y0_train, Y0_test, Y1_train, Y1_test = train_test_split(
        X, T, Y, Y0, Y1, test_size=0.2, random_state=42
    )

    # 5.2. Train CFRNet
    cfrnet = CFRNet(input_dim=X.shape[1], ipm_lambda=0.1)  # Experiment with ipm_lambda
    optimizer_cfrnet = optim.Adam(cfrnet.parameters(), lr=0.01)
    train_cfrnet(cfrnet, X_train, T_train, Y_train, optimizer_cfrnet)

    # 5.3. Evaluate CFRNet
    evaluate_cfrnet(cfrnet, X_test, T_test, Y_test, Y0_test, Y1_test)

    # 5.4. Explanation
    print("\n--- Explanation ---")
    print("This code implements the CFRNet architecture for causal inference.")
    print("CFRNet aims to learn balanced representations by minimizing a loss function that combines:")
    print("1.  Outcome prediction loss (MSE).")
    print("2.  A simplified balancing loss (difference in mean representations for treated and control groups).")
    print("The 'ipm_lambda' parameter controls the trade-off between these two objectives.")

