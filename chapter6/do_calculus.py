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
import statsmodels.formula.api as sm  # For regression

# --- 1. Simulate Data for Backdoor Criterion Example ---
def simulate_backdoor_data(n=1000, effect_t_y=2, effect_z_t=0.5, effect_z_y=1):
    """Simulates data to demonstrate the backdoor criterion."""

    Z = np.random.normal(0, 1, n)  # Confounder
    T = effect_z_t * Z + np.random.normal(0, 1, n)  # Treatment
    Y = effect_t_y * T + effect_z_y * Z + np.random.normal(0, 1, n)  # Outcome
    return pd.DataFrame({'Z': Z, 'T': T, 'Y': Y})


# --- 2. Demonstrate Naive (Biased) Effect Estimation ---
def naive_effect_estimation(data):
    """Estimates the effect of T on Y without adjusting for Z (biased)."""

    model = sm.ols('Y ~ T', data=data).fit()
    ate_naive = model.params['T']
    print("\n--- Naive Effect Estimation (Biased) ---")
    print(f"Naive ATE: {ate_naive:.2f}")
    print("This estimate is biased because it doesn't account for the confounder Z.")
    return ate_naive


# --- 3. Demonstrate Backdoor Adjustment ---
def backdoor_adjustment(data):
    """Estimates the effect of T on Y using backdoor adjustment (adjusting for Z)."""

    model = sm.ols('Y ~ T + Z', data=data).fit()
    ate_adjusted = model.params['T']
    print("\n--- Backdoor Adjustment ---")
    print(f"Adjusted ATE: {ate_adjusted:.2f}")
    print("This estimate is (unbiased) because it adjusts for the confounder Z, satisfying the backdoor criterion.")
    return ate_adjusted, model


# --- 4. Illustrate Do-Calculus Rule 3 (Simplified) ---
def illustrate_do_calculus_rule3(data):
    """Illustrates a simplified version of do-calculus Rule 3 using the simulated data."""

    print("\n--- Do-Calculus Rule 3 Illustration (Simplified) ---")
    print("Rule 3 states that if a variable Z is independent of Y given X after intervening on X, then do(Z) is irrelevant for predicting Y.")
    print("In our case, if we were to simulate 'do(T)', the relationship between Z and Y would change.")
    print("However, in this simplified simulation, we'll show the effect of adjusting vs. not adjusting.")

    # This is NOT a true 'do()' simulation, but it shows the effect of Z
    model_without_do_z = sm.ols('Y ~ T', data=data).fit()
    effect_without_do_z = model_without_do_z.params['T']
    print(f"\nEffect without adjusting for Z: {effect_without_do_z:.2f}")

    model_with_do_z = sm.ols('Y ~ T + Z', data=data).fit()
    effect_with_do_z = model_with_do_z.params['T']
    print(f"Effect after adjusting for Z (analogous to 'do(T)'): {effect_with_do_z:.2f}")  # Analogous to do(T)

    print("\nIn this simplified example, adjusting for Z is similar to what 'do(T)' would achieve (in terms of removing confounding).")


# --- 5. Main Execution ---
if __name__ == '__main__':
    np.random.seed(42)  # Consistent seed
    data = simulate_backdoor_data(effect_t_y=2, effect_z_t=0.5, effect_z_y=1)  # Control causal strengths

    naive_effect = naive_effect_estimation(data)
    adjusted_effect, regression_model = backdoor_adjustment(data)
    illustrate_do_calculus_rule3(data)  # Simplified illustration

    print("\n--- Summary ---")
    print("This code demonstrates how confounding biases the estimation of the effect of T on Y.")
    print("It shows how backdoor adjustment (analogous to do-calculus) can correct for this bias.")
    print("Specifically, it illustrates (in a simplified way) how do-calculus helps us identify and remove the influence of Z when estimating the effect of T on Y.")

