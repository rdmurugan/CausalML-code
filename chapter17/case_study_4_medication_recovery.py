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

# Case Study 4: Estimating the Effect of a New Medication on Patient Recovery
# From "Causal Inference for Machine Learning Engineers - A Practical Guide"
# by Durai Rajamanickam

# --- Simulating medication and recovery data ---

import numpy as np

np.random.seed(2025)
n = 2000
Health = np.random.normal(0, 1, size=n)  # Health score
Age = np.random.normal(50, 10, size=n)   # Age in years

# Propensity to receive medication
logit_p = 0.8 * Health - 0.03 * Age
p = 1 / (1 + np.exp(-logit_p))
Medication = np.random.binomial(1, p)

# Recovery time outcome (lower is better)
Recovery_baseline = 10 - 2 * Health + 0.05 * Age + np.random.normal(0, 1, size=n)
Treatment_effect = -1.5  # Medication reduces recovery time by 1.5 days
RecoveryTime = Recovery_baseline + Medication * Treatment_effect

# Assemble into features
X = np.stack([Health, Age], axis=1)


# --- Training DragonNet on medication data ---

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(X, Medication, RecoveryTime, test_size=0.3)

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

