import numpy as np
import pandas as pd

# Simulate data with ignorability (no confounding, for simplicity)
np.random.seed(2)
n = 1000
X = np.random.normal(0, 1, n)  # Covariates
T = np.random.binomial(1, 0.5, n)  # Random treatment assignment
Y0 = 2 * X + np.random.normal(0, 1, n)
Y1 = 2 * X + 5 + np.random.normal(0, 1, n)  # Treatment effect = 5
Y = T * Y1 + (1 - T) * Y0
data = pd.DataFrame({'X': X, 'T': T, 'Y': Y})

# Estimate ATE
ate_naive = data.loc[data['T'] == 1, 'Y'].mean() - \
            data.loc[data['T'] == 0, 'Y'].mean()
print(f"Naive ATE estimate: {ate_naive:.2f}")  # close to 5

# In this simplified case, the naive ATE is unbiased because we simulated no confounding.