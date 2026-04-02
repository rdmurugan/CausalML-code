# Case Study 2: Estimating the Effect of Online Learning on Student Performance
# From "Causal Inference for Machine Learning Engineers - A Practical Guide"
# by Durai Rajamanickam

# --- Simulating online learning data ---

import numpy as np

np.random.seed(123)
n = 3000
Motivation = np.random.normal(0, 1, n)
GPA = np.random.uniform(2.0, 4.0, n)

# Propensity to join online program
logit_p = 0.7 * Motivation + 0.5 * (GPA - 3.0)
p = 1 / (1 + np.exp(-logit_p))
OnlineProgram = np.random.binomial(1, p)

# Academic performance
Performance_baseline = 70 + 5 * Motivation + 10 * (GPA - 3.0) + np.random.normal(0, 5, n)
Treatment_effect = 5  # Online learning boosts scores by 5 points
Performance = Performance_baseline + OnlineProgram * Treatment_effect

# Assemble into features
X = np.stack([Motivation, GPA], axis=1)


# --- Training DragonNet on online learning data ---

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(X, OnlineProgram, Performance, test_size=0.3)

# Convert to tensors
import torch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
T_train_tensor = torch.tensor(T_train, dtype=torch.float32).unsqueeze(1)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1)

# Train DragonNet (reusing earlier DragonNet class)
model = DragonNet(input_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion_outcome = torch.nn.MSELoss()
criterion_propensity = torch.nn.BCELoss()

for epoch in range(100):
    optimizer.zero_grad()
    y_pred, t_pred = model(X_train_tensor, T_train_tensor)
    loss_outcome = criterion_outcome(y_pred, Y_train_tensor)
    loss_propensity = criterion_propensity(t_pred, T_train_tensor)
    loss = loss_outcome + loss_propensity
    loss.backward()
    optimizer.step()


# --- Evaluating DragonNet performance ---

with torch.no_grad():
    y0_pred, _ = model(torch.tensor(X_test, dtype=torch.float32), torch.zeros((len(X_test),1)))
    y1_pred, _ = model(torch.tensor(X_test, dtype=torch.float32), torch.ones((len(X_test),1)))

pred_effect = (y1_pred - y0_pred).squeeze().numpy()
true_effect = np.full_like(pred_effect, Treatment_effect)

# PEHE
pehe = np.sqrt(np.mean((pred_effect - true_effect)**2))

# ATE Error
true_ate = np.mean(true_effect)
pred_ate = np.mean(pred_effect)
ate_error = np.abs(pred_ate - true_ate)

print(f"PEHE: {pehe:.4f}")
print(f"ATE Error: {ate_error:.4f}")

