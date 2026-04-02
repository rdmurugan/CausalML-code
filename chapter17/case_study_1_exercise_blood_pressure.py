# Case Study 1: Estimating the Effect of an Exercise Program on Blood Pressure
# From "Causal Inference for Machine Learning Engineers - A Practical Guide"
# by Durai Rajamanickam

# --- Simulating exercise program data ---

import numpy as np

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
X = np.stack([Age, Health], axis=1)


# --- Training CFRNet on exercise program data ---

# Split data into train/test
from sklearn.model_selection import train_test_split
X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(X, Exercise, BloodPressure, test_size=0.3)

# Convert to tensors
import torch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
T_train_tensor = torch.tensor(T_train, dtype=torch.float32).unsqueeze(1)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1)

# Train CFRNet (reusing earlier CFRNet class)
model = CFRNet(input_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

for epoch in range(100):
    optimizer.zero_grad()
    pred = model(X_train_tensor, T_train_tensor)
    loss = criterion(pred, Y_train_tensor)
    loss.backward()
    optimizer.step()


# --- Evaluating CFRNet performance ---

with torch.no_grad():
    y0_pred = model(torch.tensor(X_test, dtype=torch.float32), torch.zeros((len(X_test),1)))
    y1_pred = model(torch.tensor(X_test, dtype=torch.float32), torch.ones((len(X_test),1)))

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

