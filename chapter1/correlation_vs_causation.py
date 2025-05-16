* **`chapter_1/correlation_vs_causation.py`**

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Simulate data
np.random.seed(0)
n = 1000
temperature = np.random.normal(25, 5, n)  # Average temperature 25 degrees
ice_cream_sales = 10 + 2 * temperature + np.random.normal(0, 10, n)
drowning_incidents = 5 + 1.5 * temperature + np.random.normal(0, 5, n)

# Create a DataFrame
data = pd.DataFrame({'Temperature': temperature,
                     'IceCreamSales': ice_cream_sales,
                     'DrowningIncidents': drowning_incidents})

# Calculate correlation
correlation = data[['IceCreamSales', 'DrowningIncidents']].corr()[0][1]
print(f"Correlation between ice cream sales and drowning: {correlation:.2f}")

# Visualize
plt.scatter(ice_cream_sales, drowning_incidents)
plt.xlabel("Ice Cream Sales")
plt.ylabel("Drowning Incidents")
plt.title("Correlation vs. Causation")
plt.show()

# This code generates data where both ice cream sales and drowning incidents are correlated
# with temperature (a confounder), but not directly causally related.
