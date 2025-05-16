import torch
import torch.nn as nn
import torch.optim as optim

class TARNet(nn.Module):
    def __init__(self, input_dim):
        super(TARNet, self).__init__()
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
np.random.seed(5)
n = 1000
X = np.random.rand(n, 5).astype(np.float32)
T = np.random.binomial(1, 0.5, n).astype(np.float32)
Y0 = X[:, 0] + 2 * X[:, 1] + np.random.normal(0, 1, n).astype(np.float32)
Y1 = Y0 + 3 + np.random.normal(0, 1, n).astype(np.float32)
Y = T * Y1 + (1 - T) * Y0
Y = Y.astype(np.float32)

# Train the model
model = TARNet(input_dim=5)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
x_tensor = torch.tensor(X)
t_tensor = torch.tensor(T).reshape(-1, 1)
y_tensor = torch.tensor(Y).reshape(-1, 1)

for epoch in range(10):  # Simplified training loop
    optimizer.zero_grad()
    y_pred, _, _ = model(x_tensor, t_tensor)
    loss = criterion(y_pred, y_tensor)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch}, Loss: {loss.item()}')