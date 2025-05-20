import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Define DragonNet (simplified for the case study)
class DragonNet(nn.Module):
    def __init__(self, input_dim):
        super(DragonNet, self).__init__()
        self.shared_representation = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU()
        )
        self.outcome_T1 = nn.Sequential(
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        self.outcome_T0 = nn.Sequential(
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        self.propensity_head = nn.Sequential(
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

    def forward(self, x, t):
        shared = self.shared_representation(x)
        outcome1 = self.outcome_T1(shared)
        outcome0 = self.outcome_T0(shared)
        propensity = self.propensity_head(shared)
        return outcome1, outcome0, propensity

# Simulate data
np.random.seed(123)
n = 3000
Motivation = np.random.normal(0, 1, n).astype(np.float32)
GPA = np.random.uniform(2.0, 4.0, n).astype(np.float32)

# Propensity to join online program
logit_p = 0.7 * Motivation + 0.5 * (GPA - 3.0)
p = 1 / (1 + np.exp(-logit_p))
OnlineProgram = np.random.binomial(1, p).astype(np.float32)

# Academic performance
Performance_baseline = 70 + 5 * Motivation + 10 * (GPA - 3.0) + np.random.normal(0, 5, n).astype(np.float32)
Treatment_effect = 5  # Online learning boosts scores by 5 points
Performance = Performance_baseline + OnlineProgram * Treatment_effect

# Assemble into features
X = np.stack([Motivation, GPA], axis=1).astype(np.float32)
T = OnlineProgram.astype(np.float32)
Y = Performance.astype(np.float32)


# Split data
X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(X, T, Y, test_size=0.3)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
T_train_tensor = torch.tensor(T_train, dtype=torch.float32).reshape(-1, 1)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).reshape(-1, 1)


# Train DragonNet
model = DragonNet(input_dim=2)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion_outcome = nn.MSELoss()
criterion_propensity = nn.BCELoss()

for epoch in range(100):
    optimizer.zero_grad()
    y1_pred, y0_pred, t_pred = model(X_train_tensor, T_train_tensor)
    outcome_pred = T_train_tensor * y1_pred + (1 - T_train_tensor) * y0_pred
    loss_outcome = criterion_outcome(outcome_pred, Y_train_tensor)
    loss_propensity = criterion_propensity(t_pred, T_train_tensor)
    loss = loss_outcome + loss_propensity
    loss.backward()
    optimizer.step()

# Evaluation
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    T_test_tensor = torch.tensor(T_test, dtype=torch.float32).reshape(-1, 1)
    y0_pred, y1_pred, _ = model(X_test_tensor, torch.zeros_like(T_test_tensor))
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
