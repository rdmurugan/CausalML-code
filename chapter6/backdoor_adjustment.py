import numpy as np
import pandas as pd
import statsmodels.formula.api as sm

# Simulate data
np.random.seed(4)
n = 1000
Z = np.random.normal(0, 1, n)  # Confounder
T = 0.5 * Z + np.random.normal(0, 1, n)
Y = 2 * T + Z + np.random.normal(0, 1, n)
data = pd.DataFrame({'Z': Z, 'T': T, 'Y': Y})

# Simplified Backdoor Adjustment (Linear Regression)
import statsmodels.formula.api as sm

# Naive regression
model1 = sm.ols(formula='Y ~ T', data=data).fit()
print(model1.params)

# Adjusted regression
model2 = sm.ols(formula='Y ~ T + Z', data=data).fit()
print(model2.params)

# The coefficient for T changes when we adjust for Z, showing the impact of the confounder.