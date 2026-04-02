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
# Simple DragonNet Implementation in PyTorch
# ============================================================

import torch
import torch.nn as nn

class DragonNet(nn.Module):
    def __init__(self, input_dim):
        super(DragonNet, self).__init__()
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
        self.head_propensity = nn.Sequential(
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, t):
        shared = self.shared(x)
        yt = self.head_treated(shared)
        yc = self.head_control(shared)
        prop_t = self.head_propensity(shared)
        y_pred = t * yt + (1 - t) * yc
        return y_pred, prop_t


# ============================================================
# Evaluating DragonNet with PEHE and ATE Error
# ============================================================

# Predict outcomes under treatment and control
with torch.no_grad():
    y0_pred, _ = model(X, torch.zeros_like(T))
    y1_pred, _ = model(X, torch.ones_like(T))

y0_pred = y0_pred.squeeze().numpy()
y1_pred = y1_pred.squeeze().numpy()

# Calculate treatment effects
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
# Chapter 12:  focusing on implementing the DragonNet architecture in PyTorch
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F  # For binary cross-entropy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score


# --- 1. Simulate Data ---
def simulate_treatment_effect_data(n=1000, input_dim=5, ate=2.0):
    """Simulates data with treatment effect and confounding."""

    X = np.random.rand(n, input_dim).astype(np.float32)  # Covariates
    true_treatment_effect = ate * np.ones(n).astype(np.float32) # Ensure float32

    # Simulate treatment assignment with confounding
    propensity_logits = X[:, 0] - 0.5 * X[:, 1] + 0.2 * X[:, 2]
    propensity_scores = 1 / (1 + np.exp(-propensity_logits))
    T = np.random.binomial(1, propensity_scores, size=n).astype(np.float32)

    Y0 = np.dot(X, np.random.rand(input_dim, 1)).flatten().astype(np.float32) + np.random.normal(0, 1, n).astype(np.float32) # Ensure float32
    Y1 = Y0 + true_treatment_effect + np.random.normal(0, 0.5, n).astype(np.float32) # Ensure float32
    Y = T * Y1 + (1 - T) * Y0  # Observed outcome

    # Convert to PyTorch tensors with explicit dtype
    X_tensor = torch.tensor(X, dtype=torch.float32)
    T_tensor = torch.tensor(T, dtype=torch.float32).reshape(-1, 1)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).reshape(-1, 1)
    Y0_tensor = torch.tensor(Y0, dtype=torch.float32).reshape(-1, 1)
    Y1_tensor = torch.tensor(Y1, dtype=torch.float32).reshape(-1, 1)


    return X_tensor, T_tensor, Y_tensor, Y0_tensor, Y1_tensor, propensity_scores


# --- 2. DragonNet Architecture (Simplified) ---
class DragonNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=100):
        super(DragonNet, self).__init__()
        self.shared_representation = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.outcome_T0 = nn.Linear(hidden_dim, 1)
        self.outcome_T1 = nn.Linear(hidden_dim, 1)
        self.propensity_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        shared = self.shared_representation(x)
        y0_hat = self.outcome_T0(shared)
        y1_hat = self.outcome_T1(shared)
        propensity_hat = self.propensity_head(shared)
        return y0_hat, y1_hat, propensity_hat

    def calculate_targeted_regularization(self, y0_hat, y1_hat, t, y):
        """Simplified targeted regularization (example)."""
        # Ensure the output is float32
        return torch.mean((y1_hat - y0_hat) * (t - 0.5)).to(torch.float32)


# --- 3. Training Function ---
def train_dragonnet(model, X_train, T_train, Y_train, optimizer, epochs=100, batch_size=32,
                   beta=1.0, lambda_reg=0.1):
    """Trains DragonNet."""

    # Ensure input tensors to DataLoader are float32
    X_train = X_train.to(torch.float32)
    T_train = T_train.to(torch.float32)
    Y_train = Y_train.to(torch.float32)

    dataset = TensorDataset(X_train, T_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion_outcome = nn.MSELoss()
    criterion_propensity = nn.BCELoss()
    model.train()

    for epoch in range(epochs):
        for x_batch, t_batch, y_batch in dataloader:
            optimizer.zero_grad()
            # Ensure inputs to the model are float32
            x_batch = x_batch.to(torch.float32)
            t_batch = t_batch.to(torch.float32)
            y_batch = y_batch.to(torch.float32)

            y0_hat, y1_hat, propensity_hat = model(x_batch)
            y_pred = t_batch * y1_hat + (1 - t_batch) * y0_hat

            outcome_loss = criterion_outcome(y_pred, y_batch)
            propensity_loss = criterion_propensity(propensity_hat, t_batch)
            # Ensure targeted_reg is float32
            targeted_reg = model.calculate_targeted_regularization(y0_hat.to(torch.float32), y1_hat.to(torch.float32), t_batch.to(torch.float32), y_batch.to(torch.float32))

            # Ensure all components of the loss are float32 before summing
            loss = outcome_loss.to(torch.float32) + beta * propensity_loss.to(torch.float32) + lambda_reg * targeted_reg.to(torch.float32)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}, Loss: {loss.item()}')


# --- 4. Evaluation Function ---
def evaluate_dragonnet(model, X_test, T_test, Y_test, Y0_test, Y1_test, propensity_scores_true):
    """Evaluates the trained DragonNet model."""

    model.eval()
    with torch.no_grad():
        # Ensure test data is also float32
        X_test = X_test.to(torch.float32)
        T_test = T_test.to(torch.float32)
        Y_test = Y_test.to(torch.float32)
        Y0_test = Y0_test.to(torch.float32)
        Y1_test = Y1_test.to(torch.float32)

        y0_pred_tensor, y1_pred_tensor, propensity_hat_tensor = model(X_test) # Get all outputs
        y_pred = (T_test * y1_pred_tensor + (1 - T_test) * y0_pred_tensor).squeeze().numpy()
        y0_pred = y0_pred_tensor.squeeze().numpy()
        y1_pred = y1_pred_tensor.squeeze().numpy()
        propensity_hat = propensity_hat_tensor.squeeze().numpy()
        Y_test = Y_test.squeeze().numpy()
        Y0_test = Y0_test.squeeze().numpy()
        Y1_test = Y1_test.squeeze().numpy()

        mse = mean_squared_error(Y_test, y_pred)
        ate_pred = np.mean(y1_pred - y0_pred)
        ate_true = np.mean(Y1_test - Y0_test)
        ate_error = np.abs(ate_pred - ate_true)
        pehe = calculate_pehe(Y1_test, Y0_test, y1_pred, y0_pred)
        # Convert T_test to NumPy before casting to int
        auc = roc_auc_score(T_test.cpu().squeeze().numpy().astype(int), propensity_hat) # Cast T_test to int for roc_auc_score

        print("\n--- Evaluation ---")
        print(f"MSE: {mse:.4f}")
        print(f"Predicted ATE: {ate_pred:.4f}")
        print(f"True ATE: {ate_true:.4f}")
        print(f"ATE Error: {ate_error:.4f}")
        print(f"PEHE: {pehe:.4f}")
        print(f"Propensity AUC: {auc:.4f}")  # Evaluate propensity prediction


def calculate_pehe(y1_true, y0_true, y1_pred, y0_pred):
    """Calculates the Precision in Estimation of Heterogeneous Effects (PEHE)."""
    ite_true = y1_true - y0_true
    ite_pred = y1_pred - y0_pred
    return np.sqrt(np.mean((ite_true - ite_pred) ** 2))


# --- 5. Main Execution ---
if __name__ == '__main__':
    np.random.seed(42)

    # 5.1. Generate Data
    X, T, Y, Y0, Y1, propensity_scores_true = simulate_treatment_effect_data()
    X_train, X_test, T_train, T_test, Y_train, Y_test, Y0_train, Y0_test, Y1_train, Y1_test = train_test_split(
        X, T, Y, Y0, Y1, test_size=0.2, random_state=42
    )

    # 5.2. Train DragonNet
    dragonnet = DragonNet(input_dim=X.shape[1])
    optimizer_dragonnet = optim.Adam(dragonnet.parameters(), lr=0.01)
    train_dragonnet(dragonnet, X_train, T_train, Y_train, optimizer_dragonnet)

    # 5.3. Evaluate DragonNet
    # Pass propensity_scores_true to the evaluate function if you want to compare
    # predicted propensity scores against the true ones during evaluation.
    # However, the evaluate_dragonnet function uses the predicted propensity_hat.
    # The original code passed it to evaluate_dragonnet but it wasn't used,
    # which is fine. Let's keep the signature consistent with the original call.
    evaluate_dragonnet(dragonnet, X_test, T_test, Y_test, Y0_test, Y1_test, propensity_scores_true)


    # 5.4. Explanation
    print("\n--- Explanation ---")
    print("This code implements a simplified DragonNet architecture for causal inference.")
    print("DragonNet learns a shared representation and has heads for:")
    print("1.  Outcome prediction (Y0_hat, Y1_hat).")
    print("2.  Propensity score estimation (propensity_hat).")
    print("The loss function combines outcome loss, propensity loss, and a simplified targeted regularization term.")
    print("Evaluation includes MSE, ATE error, PEHE, and AUC for propensity prediction.")

