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
import matplotlib.pyplot as plt

# --- Basic Simulation ---
np.random.seed(1)  # For reproducibility
n = 1000  # Number of data points

# Confounder: Health Consciousness (higher values = more health-conscious)
health_consciousness = np.random.normal(0, 1, n)

# Treatment: Aspirin Use (1 = uses aspirin, 0 = doesn't)
# Health-conscious people are more likely to use aspirin
aspirin_probability = 1 / (1 + np.exp(-0.5 * health_consciousness))  # Sigmoid function to get probability
aspirin = np.random.binomial(1, aspirin_probability, n)

# Outcome: Headache Relief (1 = relief, 0 = no relief)
# Health consciousness also independently affects headache relief
relief_probability = 0.2 + 0.4 * aspirin + 0.3 * health_consciousness + np.random.normal(0, 0.2, n)
relief = np.random.binomial(1, np.clip(relief_probability, 0, 1), n)  # Clip to ensure probabilities are valid

# Create Pandas DataFrame
data = pd.DataFrame({'HealthConsciousness': health_consciousness,
                     'Aspirin': aspirin,
                     'Relief': relief})

# Calculate and print the observed association (not causal!)
observed_association = data.groupby('Aspirin')['Relief'].mean()
print("\nObserved Association (Aspirin vs. Relief):")
print(observed_association)

# Visualize the data (example: histograms)
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist(data[data['Aspirin'] == 0]['Relief'], alpha=0.5, label='No Aspirin')
plt.hist(data[data['Aspirin'] == 1]['Relief'], alpha=0.5, label='Aspirin')
plt.xlabel('Relief (0 or 1)')
plt.ylabel('Frequency')
plt.title('Relief Distribution by Aspirin Use')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(data['HealthConsciousness'], data['Aspirin'], alpha=0.5)
plt.xlabel('Health Consciousness')
plt.ylabel('Aspirin Use')
plt.title('Aspirin Use vs. Health Consciousness')

plt.show()

print("\n--- Explanation ---")
print("This code simulates the effect of a confounding variable ('HealthConsciousness') on both the treatment ('Aspirin') and the outcome ('Relief').")
print("1.  'HealthConsciousness' is generated from a normal distribution.")
print("2.  'Aspirin' use is made *dependent* on 'HealthConsciousness' (health-conscious people are more likely to take it).")
print("3.  'Relief' is influenced by *both* 'Aspirin' and 'HealthConsciousness'.")
print("The plots and the 'observed_association' show that aspirin appears to be associated with relief, but this association is biased because of 'HealthConsciousness'.")
print("In a real-world scenario, if we don't account for 'HealthConsciousness', we might wrongly conclude that aspirin is more effective than it actually is.")


# --- Structured Simulation with Functions (More Advanced) ---
def simulate_confounded_data(n=1000):
    """Simulates data with a confounding variable."""

    health = np.random.normal(0, 1, n)
    aspirin_prob = 1 / (1 + np.exp(-0.5 * health))
    aspirin = np.random.binomial(1, aspirin_prob, n)
    relief_prob = 0.2 + 0.4 * aspirin + 0.3 * health + np.random.normal(0, 0.2, n)
    relief = np.random.binomial(1, np.clip(relief_prob, 0, 1), n)

    return pd.DataFrame({'Health': health, 'Aspirin': aspirin, 'Relief': relief})


def analyze_association(data):
    """Calculates and prints the association between aspirin use and relief."""

    association = data.groupby('Aspirin')['Relief'].mean()
    print("\n--- Analysis: Aspirin vs. Relief ---")
    print(association)


def visualize_confounding(data):
    """Visualizes the relationship between health consciousness and aspirin use."""

    plt.figure(figsize=(6, 6))
    plt.scatter(data['Health'], data['Aspirin'], alpha=0.5)
    plt.xlabel('Health Consciousness')
    plt.ylabel('Aspirin Use')
    plt.title('Confounding: Health vs. Aspirin')
    plt.show()


if __name__ == '__main__':  # Ensures code only runs when the script is executed directly
    data = simulate_confounded_data()
    analyze_association(data)
    visualize_confounding(data)

    print("\n--- Function-Based Explanation ---")
    print("The code is organized into functions for better readability and reusability.")
    print("1.  'simulate_confounded_data()' generates the data, encapsulating the simulation logic.")
    print("2.  'analyze_association()' calculates and prints the mean relief for aspirin users and non-users.")
    print("3.  'visualize_confounding()' creates a scatter plot to show the relationship between 'Health' and 'Aspirin'.")
    print("The 'if __name__ == '__main__':' block ensures that these functions are called only when the script is run, not when imported as a module.")

