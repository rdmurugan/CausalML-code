# MIT License
#
# Copyright (c) 2024 Durai Rajamanickam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm  # For regression
from sklearn.linear_model import LogisticRegression  # For propensity score
from sklearn.utils import resample  # For bootstrapping

import matplotlib.pyplot as plt
import seaborn as sns  # For better plots


# --- 1. Simulate Data with Confounding ---
def simulate_confounded_data(n=1000, ate=5):
    """Simulates data with a treatment, outcome, and confounder."""

    X = np.random.normal(0, 1, n)  # Confounder
    T = np.random.binomial(1, 1 / (1 + np.exp(-0.5 * X)), n)  # Treatment depends on X
    Y0 = 2 * X + np.random.normal(0, 1, n)  # Outcome without treatment
    Y1 = Y0 + ate  # Outcome with treatment (constant ATE)
    Y = T * Y1 + (1 - T) * Y0  # Observed outcome

    return pd.DataFrame({'X': X, 'T': T, 'Y': Y, 'Y0': Y0, 'Y1': Y1})  # Return potential outcomes too


# --- 2. Estimate ATE using Regression Adjustment ---
def estimate_ate_regression(data):
    """Estimates ATE using linear regression."""

    model = sm.ols('Y ~ T + X', data=data).fit()  # Adjust for X
    ate = model.params['T']  # Coefficient of T is the ATE
    print("\n--- Regression Adjustment ---")
    print(f"Estimated ATE: {ate:.2f}")

    return ate, model


# --- 3. Estimate ATE using Propensity Score Weighting (IPW) ---
def estimate_ate_ipw(data, bootstrap_iterations=1000):
    """Estimates ATE using Inverse Propensity Weighting (IPW)."""

    # 3.1. Estimate propensity scores
    propensity_model = LogisticRegression(solver='liblinear', random_state=0).fit(data[['X']], data['T'])
    propensity_scores = propensity_model.predict_proba(data[['X']])[:, 1]
    data['Propensity'] = propensity_scores

    # 3.2. Calculate IPW weights
    data['IPW'] = np.where(data['T'] == 1, 1 / data['Propensity'], 1 / (1 - data['Propensity']))

    # 3.3. Estimate ATE with weighted average
    ate_ipw = np.average(data['Y'], weights=data['IPW']) - np.average(data['Y'], weights=np.where(data['T'] == 0, data['IPW'], 0))
    print("\n--- IPW ---")
    print(f"Estimated ATE (IPW): {ate_ipw:.2f}")

    # 3.4. (Optional) Bootstrap for Confidence Intervals
    ate_estimates = []
    for _ in range(bootstrap_iterations):
        bootstrap_data = resample(data, replace=True, n_samples=len(data), random_state=np.random.randint(0, 10000))  # Resample with replacement
        ate_estimates.append(np.average(bootstrap_data['Y'], weights=bootstrap_data['IPW']) - np.average(bootstrap_data['Y'], weights=np.where(bootstrap_data['T'] == 0, bootstrap_data['IPW'], 0)))

    ci_lower = np.percentile(ate_estimates, 2.5)
    ci_upper = np.percentile(ate_estimates, 97.5)
    print(f"95% Confidence Interval: ({ci_lower:.2f}, {ci_upper:.2f})")  # More robust uncertainty

    return ate_ipw, propensity_model


# --- 4. Visualize Results ---
def visualize_results(data, regression_model, propensity_model):
    """Visualizes the data and the propensity score distribution."""

    plt.figure(figsize=(15, 5))

    # 4.1. Scatter Plot: Y vs. T, colored by X
    plt.subplot(1, 3, 1)
    plt.scatter(data['T'], data['Y'], c=data['X'], alpha=0.5)
    plt.xlabel('Treatment (T)')
    plt.ylabel('Outcome (Y)')
    plt.title('Y vs. T (colored by X)')
    plt.colorbar(label='Confounder (X)')

    # 4.2. Regression Line
    plt.subplot(1, 3, 2)
    sns.regplot(x='T', y='Y', data=data, ci=95)  # Show regression line with CI
    plt.xlabel('Treatment (T)')
    plt.ylabel('Outcome (Y)')
    plt.title('Regression of Y on T')

    # 4.3. Propensity Score Distribution
    plt.subplot(1, 3, 3)
    plt.hist(data['Propensity'], bins=20, alpha=0.5)
    plt.xlabel('Propensity Score')
    plt.ylabel('Frequency')
    plt.title('Propensity Score Distribution')

    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.show()


# --- 5. Main Execution ---
if __name__ == '__main__':
    np.random.seed(42)  # Consistent seed for the whole script
    data = simulate_confounded_data()
    ate_regression, regression_model = estimate_ate_regression(data)
    ate_ipw, propensity_model = estimate_ate_ipw(data)
    visualize_results(data, regression_model, propensity_model)

    print("\n--- Summary ---")
    print("This code simulates data with a confounder 'X' affecting both treatment 'T' and outcome 'Y'.")
    print("It then estimates the Average Treatment Effect (ATE) using two methods:")
    print("1.  Regression Adjustment: Includes 'X' as a covariate in the regression model.")
    print("2.  Inverse Propensity Weighting (IPW): Weights observations by the inverse probability of treatment.")
    print("The visualization shows the confounding effect and the distribution of propensity scores.")

