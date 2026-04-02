import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm  # For regression analysis (optional)

# Simulate data for the ice cream sales and drowning example
np.random.seed(0)  # for reproducibility
n = 1000  # number of data points

# Temperature is the common cause (confounder)
temperature = np.random.normal(25, 5, n)  # average temperature of 25 degrees Celsius with some variation

# Ice cream sales increase with temperature
ice_cream_sales = 10 + 2 * temperature + np.random.normal(0, 10, n)  # base sales + temp effect + noise

# Drowning incidents also increase with temperature
drowning_incidents = 5 + 1.5 * temperature + np.random.normal(0, 5, n)  # base incidents + temp effect + noise

# Create a Pandas DataFrame to hold the data
data = pd.DataFrame({
    'Temperature': temperature,
    'IceCreamSales': ice_cream_sales,
    'DrowningIncidents': drowning_incidents
})

# Calculate the correlation between ice cream sales and drowning incidents
correlation = data['IceCreamSales'].corr(data['DrowningIncidents'])
print(f"Correlation between ice cream sales and drowning: {correlation:.2f}")

# Visualize the data
plt.scatter(data['IceCreamSales'], data['DrowningIncidents'])
plt.xlabel('Ice Cream Sales')
plt.ylabel('Drowning Incidents')
plt.title('Correlation vs. Causation')
plt.show()

# --- Optional: Demonstrate with Regression ---
# Fit a simple linear regression model to predict drowning incidents from ice cream sales
model = sm.OLS(data['DrowningIncidents'], sm.add_constant(data['IceCreamSales'])).fit()
print(model.summary())  # Print regression summary

# Explanation of the code:
# 1. Simulation:
#    - We generate 'temperature' as the common cause.
#    - 'ice_cream_sales' and 'drowning_incidents' are both made to depend on 'temperature',
#      creating a correlation between them.
#    - Noise is added to each to make it more realistic.
# 2. DataFrame:
#    - Pandas is used to organize the data, making it easier to work with.
# 3. Correlation:
#    - The `.corr()` method calculates the Pearson correlation coefficient.
#    - A high correlation will be observed.
# 4. Visualization:
#    - A scatter plot visually shows the relationship. The plot will show that as ice cream sales increase, so do drowning incidents.
# 5. Regression (Optional):
#    - statsmodels is used to fit a linear regression.
#    - The regression output provides more detail (e.g., statistical significance), further emphasizing the apparent relationship.
#
# Key takeaway: The code simulates a scenario where a correlation exists due to a common cause.
# The regression model might suggest a predictive relationship, but it doesn't mean ice cream sales cause drowning.

