import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm  # For regression

# --- 1. Simulate Data with Confounder ---
def simulate_confounded_data(n=1000, effect_of_z=1.0, effect_of_t=2.0):
    """Simulates data with a confounder Z affecting both T and Y."""

    Z = np.random.normal(0, 1, n)  # Confounder
    T = 0.5 * Z + np.random.normal(0, 1, n)  # Treatment depends on Z
    Y = effect_of_t * T + effect_of_z * Z + np.random.normal(0, 1, n)  # Outcome depends on both
    return pd.DataFrame({'Z': Z, 'T': T, 'Y': Y})


# --- 2. Observational Analysis: Y vs. T ---
def observational_analysis(data):
    """Calculates and visualizes the observational relationship between T and Y."""

    print("\n--- Observational Analysis (Y vs. T) ---")
    print(data.groupby('T')['Y'].mean())  # Grouped means

    plt.figure(figsize=(6, 4))
    sns.regplot(x='T', y='Y', data=data, ci=95)  # Regression line with CI
    plt.xlabel('Treatment (T)')
    plt.ylabel('Outcome (Y)')
    plt.title('Observational: Y vs. T')
    plt.show()

    model = sm.ols('Y ~ T', data=data).fit()
    print(model.summary())  # Regression summary (optional)


# --- 3. Interventional Analysis: do(T) ---
def interventional_analysis(data, t_value=0):
    """Simulates the do-operator by setting T to a fixed value and analyzing Y."""

    print(f"\n--- Interventional Analysis: do(T = {t_value}) ---")

    data_do_t = data.copy()
    data_do_t['T'] = t_value  # Intervention: Set T to 't_value'

    print(f"Mean Outcome when do(T = {t_value}): {data_do_t['Y'].mean():.2f}")

    plt.figure(figsize=(6, 4))
    plt.hist(data_do_t['Y'], bins=20, alpha=0.7)
    plt.xlabel('Outcome (Y)')
    plt.ylabel('Frequency')
    plt.title(f'Interventional: Distribution of Y when do(T = {t_value})')
    plt.show()

    model_do_t = sm.ols('Y ~ 1', data=data_do_t).fit()  # Regression on constant
    print(model_do_t.summary())  # Summary (optional)


# --- 4. Comparing Observational and Interventional ---
def compare_distributions(data, data_do_0, data_do_1):
    """Compares the distributions of Y under observational and interventional settings."""

    plt.figure(figsize=(8, 5))

    sns.kdeplot(data['Y'], label='Observational', fill=True, alpha=0.3)
    sns.kdeplot(data_do_0['Y'], label='do(T=0)', fill=True, alpha=0.3)
    sns.kdeplot(data_do_1['Y'], label='do(T=1)', fill=True, alpha=0.3)

    plt.xlabel('Outcome (Y)')
    plt.ylabel('Density')
    plt.title('Comparison: Observational vs. Interventional')
    plt.legend()
    plt.show()


# --- 5. Main Execution ---
if __name__ == '__main__':
    np.random.seed(123)
    data = simulate_confounded_data(effect_of_t=2.0)  # Adjust the 'true' effect of T

    observational_analysis(data)
    interventional_analysis(data, t_value=0)
    interventional_analysis(data, t_value=1)

    data_do_0 = data.copy()
    data_do_0['T'] = 0
    data_do_1 = data.copy()
    data_do_1['T'] = 1
    compare_distributions(data, data.copy(), data_do_0, data_do_1)

    print("\n--- Explanation ---")
    print("This code simulates data with a confounder 'Z' affecting both treatment 'T' and outcome 'Y'.")
    print("It then demonstrates the difference between:")
    print("1.  Observational analysis:  Analyzing the relationship between 'T' and 'Y' as observed in the data.")
    print("2.  Interventional analysis:  Simulating the 'do(T)' operation by setting 'T' to a fixed value and observing the resulting distribution of 'Y'.")
    print("The comparison of distributions highlights how interventions can change the relationship between variables compared to passive observation.")
