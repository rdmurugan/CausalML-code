# Mini Case Study 5: Effect of Remote Work on Employee Productivity
# From "Causal Inference for Machine Learning Engineers - A Practical Guide"
# by Durai Rajamanickam

# --- Simulating remote work data ---

import numpy as np

np.random.seed(2026)
n = 1000
Discipline = np.random.normal(0, 1, size=n)

# Propensity to choose remote work
logit_p = 1.2 * Discipline
p = 1 / (1 + np.exp(-logit_p))
RemoteWork = np.random.binomial(1, p)

# Productivity outcome
Productivity_baseline = 50 + 10 * Discipline + np.random.normal(0, 5, n)
Treatment_effect = 3  # Remote work increases productivity by 3 units
Productivity = Productivity_baseline + RemoteWork * Treatment_effect

X = Discipline.reshape(-1, 1)

