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
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations

# --- 1. Simulate Data for Causal Discovery (Simplified) ---
def simulate_causal_discovery_data(n=1000):
    """Simulates data for a simple causal discovery example (A -> B -> C, A -> C)."""

    A = np.random.normal(0, 1, n)
    B = 0.8 * A + np.random.normal(0, 1, n)
    C = 0.5 * A + 0.7 * B + np.random.normal(0, 1, n)
    return pd.DataFrame({'A': A, 'B': B, 'C': C})


# --- 2. Simplified PC Algorithm (Illustrative) ---
def simplified_pc_algorithm(data, alpha=0.05):
    """A highly simplified version of the PC algorithm for illustration."""

    variables = list(data.columns)
    graph = nx.Graph()  # Undirected initially
    graph.add_nodes_from(variables)
    graph.add_edges_from(combinations(variables, 2))  # Start fully connected

    def is_independent(var1, var2, cond_set=None):
        """A very basic independence test (p-value from linear regression)."""
        if cond_set:
            formula = f'{var1} ~ ' + ' + '.join(cond_set)
        else:
            formula = f'{var1} ~ 1'  # Intercept only
        model = sm.ols(formula, data=data).fit()
        return model.f_pvalue > alpha  # Simplified: using F-test p-value

    # Very simplified: Only 0-variable conditioning
    for var1, var2 in list(graph.edges):  # Iterate over a copy to allow removal
        if is_independent(var1, var2):
            graph.remove_edge(var1, var2)

    # Very simplified: Limited orientation (collider detection)
    directed_graph = nx.DiGraph(graph)  # Convert to directed
    for a, b, c in combinations(variables, 3):
        if graph.has_edge(a, b) and graph.has_edge(b, c) and not graph.has_edge(a, c):
            if not is_independent(a, c, cond_set=[b]):
                if directed_graph.has_edge(b, a): directed_graph.remove_edge(b, a)
                if directed_graph.has_edge(b, c): directed_graph.remove_edge(b, c)
                directed_graph.add_edge(a, b)
                directed_graph.add_edge(c, b)  # Collider at B

    return directed_graph


# --- 3. Simulate Data for Instrumental Variables ---
def simulate_iv_data(n=1000, effect_t_y=2, effect_z_t=0.5, effect_u_t=0.3, effect_u_y=0.8):
    """Simulates data for an Instrumental Variables scenario (Z -> T -> Y, U -> T, U -> Y)."""

    U = np.random.normal(0, 1, n)  # Unobserved confounder
    Z = np.random.normal(0, 1, n)  # Instrument
    T = 0.5 * Z + 0.3 * U + np.random.normal(0, 1, n)
    Y = effect_t_y * T + 0.8 * U + np.random.normal(0, 1, n)
    return pd.DataFrame({'Z': Z, 'T': T, 'Y': Y, 'U': U})


# --- 4. Implement Two-Stage Least Squares (2SLS) ---
def implement_2sls(data):
    """Implements Two-Stage Least Squares (2SLS) to estimate the effect of T on Y."""

    # 4.1. First Stage: Regress T on Z
    first_stage = sm.ols('T ~ Z', data=data).fit()
    data['T_hat'] = first_stage.predict(data)  # Predicted T

    # 4.2. Second Stage: Regress Y on T_hat
    second_stage = sm.ols('Y ~ T_hat', data=data).fit()
    effect_of_t = second_stage.params['T_hat']

    print("\n--- Two-Stage Least Squares (2SLS) ---")
    print("Estimated effect of T on Y:", effect_of_t)
    return effect_of_t


# --- 5. Main Execution ---
if __name__ == '__main__':
    np.random.seed(42)

    # 5.1. Causal Discovery Example
    causal_data = simulate_causal_discovery_data()
    learned_graph = simplified_pc_algorithm(causal_data)

    plt.figure(figsize=(8, 6))
    nx.draw_circular(learned_graph, with_labels=True, node_size=2000, node_color='lightyellow', font_size=12, arrowsize=20)
    plt.title('Learned Causal Graph (Simplified PC)')
    plt.show()

    print("\n--- Causal Discovery ---")
    print("This demonstrates a simplified PC algorithm.")
    print("It learns the causal structure from data (A -> B -> C, A -> C) by testing conditional independence.")
    print("Note: This is a simplified version; the full PC algorithm is more complex.")

    # 5.2. Instrumental Variables Example
    iv_data = simulate_iv_data()
    estimated_effect_iv = implement_2sls(iv_data)
    print("\n--- Instrumental Variables ---")
    print("This demonstrates Two-Stage Least Squares (2SLS).")
    print("It estimates the effect of T on Y using an instrument Z to address unobserved confounding.")

