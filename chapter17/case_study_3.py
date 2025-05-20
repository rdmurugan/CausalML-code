import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. Simulation Function
def simulate_advertising_data(n=2000):
    """Simulates data for the advertising and sales case study."""

    np.random.seed(2025)
    Seasonality = np.random.choice([0, 1], size=n, p=[0.7, 0.3]).astype(np.float32)  # 0 = regular, 1 = holiday
    Rating = np.random.uniform(1.0, 5.0, size=n).astype(np.float32)  # Product rating
    propensity = 1 / (1 + np.exp(-(0.6 * Seasonality + 0.5 * (Rating - 3.0))))
    AdSpend = np.random.binomial(1, propensity, n).astype(np.float32)
    Sales_baseline = 200 + 30 * Seasonality + 50 * (Rating - 3.0) + np.random.normal(0, 20, n).astype(np.float32)
    Treatment_effect = 40  # Effect of advertising
    Sales = Sales_baseline + AdSpend * Treatment_effect + np.random.normal(0, 10, n).astype(np.float32)

    return torch.tensor(np.stack([Seasonality, Rating], axis=1)), torch.tensor(AdSpend).reshape(-1, 1), torch.tensor(Sales).reshape(-1, 1), Treatment_effect

# 2. CFRNet Model (Reusing the same architecture as in Case Study 1)
class CFRNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.shared_representation = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.outcome_T0 = nn.Linear(hidden_dim, 1)
        self.outcome_T1 = nn.Linear(hidden_dim, 1)

    def forward(self, x, t):
        shared = self.shared_representation(x)
        y0_hat = self.outcome_T0(shared)
        y1_hat = self.outcome_T1(shared)
        y_hat = t * y1_hat + (1 - t) * y0_hat
        return y_hat, y0_hat, y1_hat

# 3. Training Function (Reusing the same training function as in Case Study 1)
def train_cfrnet(model, X_train, T_train, Y_train, epochs=100, batch_size=32, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    dataset = TensorDataset(X_train, T_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for x_batch, t_batch, y_batch in dataloader:
            optimizer.zero_grad()
            y_pred, _, _ = model(x_batch, t_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# 4. Evaluation Function (Reusing the same evaluation function as in Case Study 1)
def evaluate_cfrnet(model, X_test, T_test, Y_test, treatment_effect_true):
    model.eval()
    with torch.no_grad():
        y_pred, y0_pred, y1_pred = model(X_test, T_test)
        y0_pred = y0_pred.squeeze().numpy()
        y1_pred = y1_pred.squeeze().numpy()
        Y_test = Y_test.squeeze().numpy()

        ate_pred = np.mean(y1_pred - y0_pred)
        ate_error = np.abs(ate_pred - treatment_effect_true)
        mse = mean_squared_error(Y_test, y_pred)
        pehe = np.mean((y1_pred - y0_pred - treatment_effect_true) ** 2)

        print("\n--- Evaluation ---")
        print(f"ATE Error: {ate_error:.4f}")
        print(f"PEHE: {pehe:.4f}")
        print(f"MSE: {mse:.4f}")

# 5. Main Execution
if __name__ == '__main__':
    X, T, Y, true_effect = simulate_advertising_data()
    X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(X, T, Y, test_size=0.2, random_state=42)

    model = CFRNet(input_dim=2)
    train_cfrnet(model, X_train, T_train, Y_train)
    evaluate_cfrnet(model, X_test, T_test, Y_test, true_effect)

    print("\n--- Summary ---")
    print("This code simulates the effect of advertising spend on sales, accounting for confounding by seasonality and product rating.")
    print("CFRNet is used to estimate the treatment effect.")
    print("Evaluation metrics are ATE Error and PEHE.")
