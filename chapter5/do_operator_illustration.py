import numpy as np
import pandas as pd

# Simulate data
np.random.seed(3)
n = 1000
Z = np.random.normal(0, 1, n)  # Confounder
T = 0.5 * Z + np.random.normal(0, 1, n)
Y = 2 * T + Z + np.random.normal(0, 1, n)
data = pd.DataFrame({'Z': Z, 'T': T, 'Y': Y})

# Observational: Y | T
observational_effect = data.groupby('T')['Y'].mean()
print("Observational effect:\n", observational_effect)

# Simplified "do(T)": set T to a fixed value (e.g., 0 or 1)
T_do_0 = data.copy()
T_do_0['T'] = 0
do_0_effect = T_do_0['Y'].mean()

T_do_1 = data.copy()
T_do_1['T'] = 1
do_1_effect = T_do_1['Y'].mean()

print(f"Simplified do(T=0) effect: {do_0_effect:.2f}")
print(f"Simplified do(T=1) effect: {do_1_effect:.2f}")

# This VERY SIMPLIFIED do-operator example shows how forcing T to a value changes the average of Y,
# compared to just observing Y at different values of T.  A true 'do' requires more complex adjustment.