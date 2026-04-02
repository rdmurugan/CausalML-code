# Causal Inference for Machine Learning Engineers - A Practical Guide

## Code Examples

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Author](https://img.shields.io/badge/Author-Durai_Rajamanickam-blue)](https://www.linkedin.com/in/durairajamanickam/)

This repository contains the Python code examples that accompany the book:

**Causal Inference for Machine Learning Engineers - A Practical Guide** by Durai Rajamanickam

This book bridges the gap between causal inference and deep learning, providing a practical introduction to causal thinking and demonstrating how deep learning architectures can be adapted for causal inference tasks.

## Key Features

- **Comprehensive Coverage:** Code examples illustrate key concepts from basic causal reasoning to advanced deep learning models across most chapters of the book.
- **Practical Focus:** The code emphasizes practical implementation and adaptation of techniques, enabling readers to apply causal deep learning to real-world problems.
- **Clear Organization:** Code is structured by chapter, making it easy to navigate and find relevant examples.
- **Beginner-Friendly:** The examples are designed to be accessible and well-documented, aligning with the book's goal of introducing the subject to a broad audience.

## Repository Structure

```
CausalML-code/
├── README.md
├── requirements.txt
├── chapter1/
│   └── correlation_vs_causation.py
├── chapter2/
│   └── confounding_simulation.py
├── chapter3/
│   └── ate_estimation.py
├── chapter4/
│   └── causal_graph_visualization.py
├── chapter5/
│   └── intervention_and_counterfactuals.py
├── chapter6/
│   └── do_calculus.py
├── chapter7/
│   └── backdoor_frontdoor_criterion.py
├── chapter8/
│   └── instrumental_variables_mediation.py
├── chapter9/
│   └── tarnet_cfrnet_dml.py
├── chapter10/
│   └── evaluation_metrics.py
├── chapter11/
│   └── cfrnet_example.py
├── chapter12/
│   └── dragonnet_example.py
├── chapter13/
│   └── evaluation_pehe_ate_error.py
├── chapter14/
│   └── pc_instrument_variables.py
└── chapter17/
    ├── case_study_1_exercise_blood_pressure.py
    ├── case_study_2_online_learning.py
    ├── case_study_3_advertising_sales.py
    ├── case_study_4_medication_recovery.py
    └── case_study_5_remote_work.py
```

## Chapter Overview

### Part I: Foundations of Causal Thinking (Chapters 1-6)

| Chapter | Title | Code Topics |
|---------|-------|-------------|
| 1 | Introduction to Causal Thinking | Correlation vs. causation simulation (ice cream sales & drowning), regression analysis |
| 2 | Treatments, Outcomes, and Confounding | Confounding simulation (aspirin & health consciousness), spurious associations |
| 3 | Causal Estimation Basics | ATE estimation, regression adjustment, Inverse Propensity Weighting (IPW), bootstrapping |
| 4 | Causal Graphs: Structure and Assumptions | Directed graphs with NetworkX, adjacency matrices, chains, forks, colliders |
| 5 | Interventions and Counterfactuals | do-operator simulation, observational vs. interventional analysis |
| 6 | Introduction to Do-Calculus | Backdoor criterion, do-calculus rules, naive vs. adjusted effect estimation |

### Part II: Causal Identification and Estimation (Chapters 7-8)

| Chapter | Title | Code Topics |
|---------|-------|-------------|
| 7 | Backdoor and Frontdoor Criteria | Backdoor adjustment, frontdoor adjustment with mediators, regression-based estimation |
| 8 | Advanced Causal Inference Methods | Instrumental Variables (2SLS), mediation analysis (direct/indirect/total effects) |

### Part III: Deep Learning for Causal Inference (Chapters 9-12)

| Chapter | Title | Code Topics |
|---------|-------|-------------|
| 9 | Causal Inference Meets Deep Learning | Double Machine Learning (DML), TARNet, CFRNet (PyTorch) |
| 10 | Simulating Causal Data and Evaluation Metrics | Causal data simulation, PEHE, ATE error, policy risk |
| 11 | Balancing Representations with Causal Deep Learning (CFRNet) | CFRNet architecture, IPM balancing loss, training and evaluation |
| 12 | Propensity Scores in Causal Deep Learning | DragonNet architecture, propensity score head, targeted regularization |

### Part IV: Evaluation, Advanced Topics, and Applications (Chapters 13-17)

| Chapter | Title | Code Topics |
|---------|-------|-------------|
| 13 | Evaluating Causal Models Without Counterfactuals | Model evaluation with PEHE/ATE, Linear Regression vs. Random Forest comparison |
| 14 | Advanced Topics in Causal Inference | Causal discovery (simplified PC algorithm), Instrumental Variables (2SLS) |
| 15 | Assumptions and Real-World Challenges | Key assumptions (SUTVA, positivity, ignorability) — conceptual chapter, no code |
| 16 | Summary of Key Concepts | Recap and emerging directions — conceptual chapter, no code |
| 17 | Case Studies | 5 applied case studies using CFRNet and DragonNet |

### Case Studies (Chapter 17)

| Case Study | Topic | Method |
|------------|-------|--------|
| 1 | Effect of an Exercise Program on Blood Pressure | CFRNet |
| 2 | Effect of Online Learning on Student Performance | DragonNet |
| 3 | Effect of Advertising on Sales | CFRNet |
| 4 | Effect of a New Medication on Patient Recovery | DragonNet |
| 5 | Effect of Remote Work on Employee Productivity | CFRNet |

## Key Methods Implemented

- **Regression Adjustment** for confounding control
- **Inverse Propensity Weighting (IPW)** with bootstrapped confidence intervals
- **Causal Graphs** using NetworkX (chains, forks, colliders)
- **Do-Calculus** and the backdoor/frontdoor criteria
- **Instrumental Variables** via Two-Stage Least Squares (2SLS)
- **Mediation Analysis** (direct, indirect, and total effects)
- **Double Machine Learning (DML)** with cross-fitting
- **TARNet** (Treatment-Agnostic Representation Networks)
- **CFRNet** (Counterfactual Regression Networks) with IPM balancing
- **DragonNet** with propensity score estimation and targeted regularization
- **Causal Discovery** (simplified PC algorithm)
- **Evaluation Metrics:** PEHE, ATE Error, Policy Risk

## Requirements

- Python 3.8+
- pip

**Python Libraries:**

- numpy
- pandas
- matplotlib
- seaborn
- statsmodels
- torch (PyTorch)
- scikit-learn
- networkx

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/rdmurugan/CausalML-code.git
   cd CausalML-code
   ```

2. **Install the required libraries:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the code examples, navigate to the directory of the chapter you are interested in and execute the Python script:

```bash
cd chapter1
python correlation_vs_causation.py
```

```bash
cd chapter9
python tarnet_cfrnet_dml.py
```

## Disclaimer

These code examples are provided for educational purposes to illustrate the concepts discussed in the book. They are simplified for clarity and may not be optimized for performance or suitable for production environments. Always adapt and validate the code for your specific use case. Please refer to the book for detailed explanations, theoretical background, and best practices.

## Contributions

Contributions to this repository are welcome! If you find a bug, have a suggestion, or want to add an example, please:

1. Fork the repository.
2. Create a new branch for your changes.
3. Commit your changes with clear and descriptive commit messages.
4. Push your changes to your fork.
5. Submit a pull request.

Please ensure your code meets basic Python style guidelines (PEP 8).

## License

This code is released under the [MIT License](https://opensource.org/licenses/MIT). You are free to use, modify, and distribute it for both commercial and non-commercial purposes.

## Author

**Durai Rajamanickam**

AI and Data Science Leader | Causal Inference Researcher

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Durai_Rajamanickam-blue)](https://www.linkedin.com/in/durairajamanickam/)
