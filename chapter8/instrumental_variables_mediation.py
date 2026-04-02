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
import pandas as pd
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Simulate Data for Instrumental Variables ---
def simulate_iv_data(n=1000, effect_t_y=2, effect_z_t=0.5, effect_u_t=0.3, effect_u_y=0.8):
    """Simulates data for an Instrumental Variables scenario (Z -> T -> Y, U -> T, U -> Y).

    Args:
        n: Number of samples.
        effect_t_y: True causal effect of T on Y.
        effect_z_t: Effect of instrument Z on treatment T.
        effect_u_t: Effect of unobserved confounder U on T.
        effect_u_y: Effect of U on Y.

    Returns:
        Pandas DataFrame with Z, T, Y, and U.
    """

    U = np.random.normal(0, 1, n)  # Unobserved confounder
    Z = np.random.normal(0, 1, n)  # Instrument
    T = effect_z_t * Z + effect_u_t * U + np.random.normal(0, 1, n)
    Y = effect_t_y * T + effect_u_y * U + np.random.normal(0, 1, n)
    return pd.DataFrame({'Z': Z, 'T': T, 'Y': Y, 'U': U})


# --- 2. Implement Two-Stage Least Squares (2SLS) ---
def implement_2sls(data, plot_results=True):
    """Implements Two-Stage Least Squares (2SLS) to estimate the effect of T on Y.

    Args:
        data: DataFrame with Z, T, Y.
        plot_results: Whether to plot the stages.

    Returns:
        Estimated effect of T on Y.
    """

    # 2.1. First Stage: Regress T on Z
    first_stage = sm.ols('T ~ Z', data=data).fit()
    data['T_hat'] = first_stage.predict(data)  # Predicted T

    # 2.2. Second Stage: Regress Y on T_hat
    second_stage = sm.ols('Y ~ T_hat', data=data).fit()
    effect_of_t = second_stage.params['T_hat']

    print("\n--- Two-Stage Least Squares (2SLS) ---")
    print("Estimated effect of T on Y:", effect_of_t)

    if plot_results:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.regplot(x='Z', y='T', data=data, ci=95)
        plt.title('First Stage: T vs. Z')
        plt.xlabel('Instrument (Z)')
        plt.ylabel('Treatment (T)')

        plt.subplot(1, 2, 2)
        sns.regplot(x='T_hat', y='Y', data=data, ci=95)
        plt.title('Second Stage: Y vs. Predicted T')
        plt.xlabel('Predicted Treatment (T_hat)')
        plt.ylabel('Outcome (Y)')

        plt.tight_layout()
        plt.show()

    return effect_of_t


# --- 3. Simulate Data for Mediation Analysis ---
def simulate_mediation_data(n=1000, effect_t_m=0.7, effect_t_y_direct=0.5, effect_m_y=0.8):
    """Simulates data for a mediation analysis scenario (T -> M -> Y, T -> Y).

    Args:
        n: Number of samples.
        effect_t_m: Effect of T on mediator M.
        effect_t_y_direct: Direct effect of T on Y.
        effect_m_y: Effect of M on Y.

    Returns:
        Pandas DataFrame with T, M, and Y.
    """

    T = np.random.normal(0, 1, n)
    M = effect_t_m * T + np.random.normal(0, 1, n)  # Mediator
    Y = effect_t_y_direct * T + effect_m_y * M + np.random.normal(0, 1, n)
    return pd.DataFrame({'T': T, 'M': M, 'Y': Y})


# --- 4. Implement Mediation Analysis (Linear Case) ---
def implement_mediation_analysis(data, plot_results=True):
    """Implements a simplified mediation analysis (for linear case).

    Args:
        data: DataFrame with T, M, Y.
        plot_results: Whether to plot the relationships.

    Returns:
        Direct, indirect, and total effects.
    """

    # 4.1. Estimate T -> M
    model_tm = sm.ols('M ~ T', data=data).fit()
    effect_t_to_m = model_tm.params['T']
    print("\n--- Mediation Analysis ---")
    print("Effect of T on M:", effect_t_to_m)

    # 4.2. Estimate T, M -> Y
    model_tmy = sm.ols('Y ~ T + M', data=data).fit()
    effect_m_to_y = model_tmy.params['M']
    effect_t_to_y_direct = model_tmy.params['T']
    print("Effect of M on Y (adjusted for T):", effect_m_to_y)
    print("Direct effect of T on Y:", effect_t_to_y_direct)

    # 4.3. Calculate indirect effect
    effect_t_to_y_indirect = effect_t_to_m * effect_m_to_y
    effect_t_to_y_total = effect_t_to_y_direct + effect_t_to_y_indirect
    print("Indirect effect of T on Y (through M):", effect_t_to_y_indirect)
    print("Total effect of T on Y:", effect_t_to_y_total)

    if plot_results:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        sns.regplot(x='T', y='M', data=data, ci=95)
        plt.title('T -> M')
        plt.xlabel('Treatment (T)')
        plt.ylabel('Mediator (M)')

        plt.subplot(1, 3, 2)
        sns.regplot(x='T', y='Y', data=data, ci=95)
        plt.title('Direct Effect of T on Y')
        plt.xlabel('Treatment (T)')
        plt.ylabel('Outcome (Y)')

        plt.subplot(1, 3, 3)
        sns.regplot(x='M', y='Y', data=data, ci=95)
        plt.title('M -> Y (adjusted for T)')
        plt.xlabel('Mediator (M)')
        plt.ylabel('Outcome (Y)')

        plt.tight_layout()
        plt.show()

    return effect_t_to_y_direct, effect_t_to_y_indirect, effect_t_to_y_total


# --- 5. Main Execution ---
if __name__ == '__main__':
    np.random.seed(42)  # Consistent seed

    # Instrumental Variables Example
    iv_data = simulate_iv_data(effect_t_y=2, effect_z_t=0.5, effect_u_t=0.3, effect_u_y=0.8)
    estimated_effect_iv = implement_2sls(iv_data)
    print("\nIn the IV example, Z is the instrument, U is the unobserved confounder.")
    print("2SLS attempts to recover the true effect of T on Y (which is 2 in this simulation).")

    # Mediation Analysis Example
    mediation_data = simulate_mediation_data(effect_t_m=0.7, effect_t_y_direct=0.5, effect_m_y=0.8)
    direct_effect, indirect_effect, total_effect = implement_mediation_analysis(mediation_data)
    print("\nIn the Mediation example, M is the mediator.")
    print("The code decomposes the total effect of T on Y into direct and indirect effects (through M).")

