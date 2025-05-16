import numpy as np
import pandas as pd

# Simulate data
np.random.seed(1)
n = 1000

# Confounder: Health Consciousness
health_consciousness = np.random.normal(0, 1, n)

# Treatment: Aspirin Use
# Health-conscious people are more likely to take aspirin
aspirin = np.random.binomial(1,
                            1 / (1 + np.exp(-0.5 * health_consciousness)), n)

# Outcome: Headache Relief
# Health consciousness also independently affects headache relief
relief_probability = 0.2 + 0.4 * aspirin + \
                      0.3 * health_consciousness + \
                      np.random.normal(0, 0.2, n)
relief = np.random.binomial(1, np.clip(relief_probability, 0, 1), n)

data = pd.DataFrame({'HealthConsciousness': health_consciousness,
                     'Aspirin': aspirin,
                     'Relief': relief})

# Showcasing how 'relief' is influenced by both 'aspirin' and 'health_consciousness'
print(data.head())