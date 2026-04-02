# Case Study 3: Estimating the Effect of Advertising on Sales
# From "Causal Inference for Machine Learning Engineers - A Practical Guide"
# by Durai Rajamanickam

# --- Simulating advertising and sales data ---

import numpy as np

np.random.seed(2024)
n = 2500
Seasonality = np.random.choice([0, 1], size=n, p=[0.7, 0.3])  # 0 = regular, 1 = holiday season
Rating = np.random.uniform(1.0, 5.0, size=n)  # Product ratings

# Propensity to spend more on ads
logit_p = 0.6 * Seasonality + 0.5 * (Rating - 3.0)
p = 1 / (1 + np.exp(-logit_p))
AdSpend = np.random.binomial(1, p)

# Sales outcome
Sales_baseline = 200 + 30 * Seasonality + 50 * (Rating - 3.0) + np.random.normal(0, 20, n)
Treatment_effect = 40  # Advertising increases sales by 40 units
Sales = Sales_baseline + AdSpend * Treatment_effect

# Assemble into features
X = np.stack([Seasonality, Rating], axis=1)


# --- Training CFRNet on advertising data ---

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(X, AdSpend, Sales, test_size=0.3)

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

