import numpy as np
import pandas as pd
import statsmodels.formula.api as sm

# Simulate data with an Instrument
np.random.seed(7)
n = 1000
Z = np.random.normal(0, 1, n)  # Instrument
U = np.random.normal(0, 1, n)  # Unobserved confounder
T = 0.5 * Z + 0.5 * U + np.random.normal(0, 1, n)
Y = 2 * T + 0.5 * U + np.random.normal(0, 1, n)
data = pd.DataFrame({'Z': Z, 'T': T, 'Y': Y})

# First Stage
model1 = sm.ols(formula='T ~ Z', data=data).fit()
data['T_hat'] = model1.predict(data)

# Second Stage
model2 = sm.ols(formula='Y ~ T_hat', data=data).fit()
print("2SLS estimate:", model2.params['T_hat'])