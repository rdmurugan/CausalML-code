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
def simulate_remote_work_data(n=1000):
    """Simulates data for the remote work and productivity case study."""

    np.random.seed(2026)
    Discipline = np.random.normal(0, 1, size=n).astype(np.float32)
    propensity = 1 / (1 + np.exp(-1.2 * Discipline))
    RemoteWork = np.random.binomial(1, propensity, size=n).astype(np.float32)
    Productivity_baseline = 50 + 10 * Discipline + np.random.normal(0, 5, size=n).astype(np.float32)
    Treatment_effect = 3  # Effect of remote work
    Productivity = Productivity_baseline + RemoteWork * Treatment_effect + np.random.normal(0, 2, size=n).astype(np.float32)

    return torch.tensor(Discipline).reshape(-1, 1), torch.tensor(RemoteWork).reshape(-1, 1), torch.tensor(Productivity).reshape(-1, 1), Treatment_effect

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

# 4. Evaluation Function (Reusing the evaluation function from Case Study 1)
def evaluate_cfrnet(model, X_test, T_test, Y_test, treatment_effect_true):
    model.eval()
    with torch.no_grad():
        y_
