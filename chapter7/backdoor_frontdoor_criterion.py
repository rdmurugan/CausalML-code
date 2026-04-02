# ============================================================
# Backdoor Adjustment with Regression
# ============================================================

import numpy as np
import pandas as pd
import statsmodels.formula.api as sm  # For regression

# Simulate data
np.random.seed(10)
n = 1000
Z = np.random.normal(0, 1, n)  # Confounder
T = 0.5 * Z + np.random.normal(0, 1, n)
Y = 2 * T + Z + np.random.normal(0, 1, n)
data = pd.DataFrame({'Z': Z, 'T': T, 'Y': Y})

# 1. Naive Regression (biased)
model_naive = sm.ols("Y ~ T", data=data).fit()
print("Naive T coefficient:", model_naive.params['T'])

# 2. Regression Adjustment (unbiased, if Z blocks all backdoor paths)
model_adjusted = sm.ols("Y ~ T + Z", data=data).fit()
print("Adjusted T coefficient:", model_adjusted.params['T'])

# Explanation:
#   -   The coefficient of T in `model_adjusted` is our estimate of the causal effect of T on Y, 
#       *assuming* Z satisfies the backdoor criterion.
#   -   Statsmodels handles the summation implicitly through the regression.


# ============================================================
# Chapter 7: Focusing on demonstrating the backdoor and frontdoor criteria with simulated data and regression.
# ============================================================

import numpy as np
import pandas as pd
import statsmodels.formula.api as sm  # For regression
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Simulate Data for Backdoor Criterion ---
def simulate_backdoor_data(n=1000, effect_t_y=2, effect_z_t=0.5, effect_z_y=1):
    """Simulates data to demonstrate the backdoor criterion (Z -> T -> Y, Z -> Y)."""

    Z = np.random.normal(0, 1, n)  # Confounder
    T = 0.5 * Z + np.random.normal(0, 1, n)  # Treatment depends on Z
    Y = effect_t_y * T + effect_z_y * Z + np.random.normal(0, 1, n)
    return pd.DataFrame({'Z': Z, 'T': T, 'Y': Y})


# --- 2. Simulate Data for Frontdoor Criterion ---
def simulate_frontdoor_data(n=1000, effect_t_m=1.0, effect_m_y=1.5, effect_u_t=0.5, effect_u_y=0.8):
    """Simulates data to demonstrate the frontdoor criterion (T -> M -> Y, U -> T, U -> Y)."""

    U = np.random.normal(0, 1, n)  # Unobserved confounder
    T = effect_u_t * U + np.random.normal(0, 1, n)
    M = effect_t_m * T + np.random.normal(0, 1, n)  # Mediator
    Y = effect_m_y * M + effect_u_y * U + np.random.normal(0, 1, n)
    return pd.DataFrame({'U': U, 'T': T, 'M': M, 'Y': Y})


# --- 3. Demonstrate Backdoor Adjustment ---
def demonstrate_backdoor_adjustment(data):
    """Demonstrates backdoor adjustment using regression."""

    # 3.1. Naive regression (biased)
    model_naive = sm.ols('Y ~ T', data=data).fit()
    ate_naive = model_naive.params['T']
    print("\n--- Backdoor Criterion ---")
    print("Naive ATE (biased):", ate_naive)

    # 3.2. Adjusted regression (unbiased)
    model_adjusted = sm.ols('Y ~ T + Z', data=data).fit()
    ate_adjusted = model_adjusted.params['T']
    print("Adjusted ATE (unbiased):", ate_adjusted)

    # 3.3. Visualization
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    sns.regplot(x='T', y='Y', data=data, ci=95)
    plt.title('Naive: Y vs. T')

    plt.subplot(1, 2, 2)
    sns.regplot(x='T', y='Y', data=data, ci=95)
    sns.regplot(x='Z', y='Y', data=data, ci=None, color='red')  # Show Z's influence
    plt.title('Adjusted: Y vs. T and Z')

    plt.tight_layout()
    plt.show()

    print("\nBackdoor adjustment corrects for confounding by including Z in the regression.")


# --- 4. Demonstrate Frontdoor Adjustment (Simplified) ---
def demonstrate_frontdoor_adjustment(data):
    """Demonstrates a simplified version of frontdoor adjustment using regression."""

    # 4.1. Estimate T -> M
    model_tm = sm.ols('M ~ T', data=data).fit()
    effect_t_to_m = model_tm.params['T']
    print("\n--- Frontdoor Criterion ---")
    print("Effect of T on M:", effect_t_to_m)

    # 4.2. Estimate M, T -> Y
    model_my = sm.ols('Y ~ M + T', data=data).fit()
    effect_m_to_y = model_my.params['M']
    print("Effect of M on Y (adjusted for T):", effect_m_to_y)

    # 4.3. Simplified frontdoor estimate (product of coefficients)
    frontdoor_estimate = effect_t_to_m * effect_m_to_y
    print("Simplified Frontdoor Estimate:", frontdoor_estimate)

    print("\nFrontdoor adjustment uses the mediator M to estimate the effect of T on Y, even with unobserved confounding U.")

    # 4.4. Visualization (Simplified)
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    sns.regplot(x='T', y='M', data=data, ci=95)
    plt.title('T vs. M')

    plt.subplot(1, 2, 2)
    sns.regplot(x='M', y='Y', data=data, ci=95)
    sns.regplot(x='T', y='Y', data=data, ci=None, color='red')  # Show T's direct influence
    plt.title('M vs. Y (adjusted for T)')

    plt.tight_layout()
    plt.show()


# --- 5. Main Execution ---
if __name__ == '__main__':
    np.random.seed(123)

    # Backdoor Example
    backdoor_data = simulate_backdoor_data(effect_t_y=2, effect_z_t=0.5, effect_z_y=1)
    demonstrate_backdoor_adjustment(backdoor_data)

    # Frontdoor Example
    frontdoor_data = simulate_frontdoor_data(effect_t_m=1.0, effect_m_y=1.5, effect_u_t=0.5, effect_u_y=0.8)
    demonstrate_frontdoor_adjustment(frontdoor_data)

