import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Define CFRNet (simplified for the case study)
class CFRNet(nn.Module):
    def __init__(self, input_dim):
        super(CFRNet, self).__init__()
        self.shared_representation = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU()
        )
        self.outcome_T1 = nn.Linear(10, 1)
        self.outcome_T0 = nn.Linear(10, 1)

    def forward(self, x, t):
        shared = self.shared_representation(x)
        outcome1 = self.outcome_T1(shared)
        outcome0 = self.outcome_T0(shared)
        return t * outcome1 + (1 - t) * outcome0, outcome0, outcome1

# Simulate data
np.random.seed(42)
n = 2000
Age = np.random.normal(50, 10, n)  # Mean age 50
Health = np.random.normal(0, 1, n)  # Baseline health score

# Propensity to participate based on health and age
logit_p = 0.05 * Age + 0.8 * Health
p = 1 / (1 + np.exp(-logit_p))
Exercise = np.random.binomial(1, p)

# Blood pressure outcome
BP_baseline = 140 - 0.3 * Health - 0.2 * Age + np.random.normal(0, 5, n)
Treatment_effect = -5  # Exercise reduces BP by 5 units
BloodPressure = BP_baseline + Exercise * Treatment_effect

# Assemble into features
X = np.stack([Age, Health], axis=1).astype(np.float32)
T = Exercise.astype(np.float32)
Y = BloodPressure.astype(np.float32)


# Split data into train / test
X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(X, T, Y, test_size=0.3)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
T_train_tensor = torch.tensor(T_train, dtype=torch.float32).reshape(-1, 1)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).reshape(-1, 1)


# Train CFRNet
model = CFRNet(input_dim=2)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(100):
    optimizer.zero_grad()
    pred, _, _ = model(X_train_tensor, T_train_tensor)
    loss = criterion(pred, Y_train_tensor)
    loss.backward()
    optimizer.step()

# Evaluation
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    T_test_tensor = torch.tensor(T_test, dtype=torch.float32).reshape(-1, 1)
    y0_pred, _, _ = model(X_test_tensor, torch.zeros_like(T_test_tensor))
    y1_pred, _, _ = model(X_test_tensor, torch.ones_like(T_test_tensor))

    pred_effect = (y1_pred - y0_pred).squeeze().numpy()
    true_effect = np.full_like(pred_effect, Treatment_effect)

    # PEHE
    pehe = np.sqrt(np.mean((pred_effect - true_effect) ** 2))

    # ATE Error
    true_ate = np.mean(true_effect)
    pred_ate = np.mean(pred_effect)
    ate_error = np.abs(pred_ate - true_ate)

    print(f"PEHE: {pehe:.4f}")
    print(f"ATE Error: {ate_error:.4f}")
