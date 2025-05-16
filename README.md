#   Causal Deep Learning: A Beginner's Journey - Code Examples

[![License:   MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Author](https://img.shields.io/badge/Author-Durai_Rajamanickam-blue)](https://www.linkedin.com/in/durairajamanickam/)
This   repository contains the Python code examples that accompany the book:

**Causal   Deep Learning: A Beginner's Journey** by Durai Rajamanickam

This   book bridges the gap between causal inference and deep learning, providing   a practical introduction to causal thinking and demonstrating how deep   learning architectures can be adapted for causal inference tasks.

##   ✨ Key Features

* **Comprehensive   Coverage:** Code examples are provided for most chapters of the book,   illustrating key concepts from basic causal reasoning to advanced deep   learning models.
* **Practical   Focus:** The code emphasizes practical implementation and adaptation of   techniques, enabling readers to apply causal deep learning to real-world   problems.
* **Clear   Organization:** Code is structured by chapter, making it easy to navigate   and find relevant examples.
* **Beginner-Friendly:** The examples are designed to be accessible and well-documented, aligning   with the book's goal of introducing the subject to a broad audience.

##   📂 Organization

The   code is organized into directories corresponding to the book's chapters:

CausalDL-Code/
├──   README.md         &lt;- You are here!
├──   chapter_1/       &lt;- Introduction to Causal Thinking
│   └──   correlation_vs_causation.py
├──   chapter_2/       &lt;- Understanding Treatments, Outcomes, and Confounding
│   └──   confounding_simulation.py
├──   chapter_3/       &lt;- Causal Estimation Basics
│   └──   ate_estimation.py
├──   chapter_4/       &lt;- Causal Graphs
│   └──   (causal_graph_visualization.py) # Optional graph visualization
├──   chapter_5/       &lt;- Interventions and Counterfactuals
│   └──   do_operator_illustration.py
├──   chapter_6/       &lt;- Introduction to Do-Calculus
│   └──   backdoor_adjustment.py
├──   chapter_7/       &lt;- Causal Inference Meets Deep Learning
│   └──   tarnet_example.py
├──   chapter_8/       &lt;- Simulating Causal Data and Evaluation Metrics
│   └──   data_simulation_and_metrics.py
├──   chapter_9/       &lt;- Balancing Representations - CFRNet
│   └──   cfrnet_loss_example.py
├──   chapter_10/      &lt;- Learning Propensity Scores and DragonNet
│   └──   propensity_score_loss.py
├──   chapter_11/      &lt;- Evaluating Causal Models Without Counterfactuals
│   └──   pehe_calculation.py
├──   chapter_13/      &lt;- Advanced Topics in Causal Inference
│   └──   instrumental_variables.py
├──   chapter_14/      &lt;- Case Studies
│   ├──   case_study_1.py
│   ├──   case_study_2.py
│   ├──   ... (etc.)
└──   utils/          &lt;- Helper functions (e.g., plotting)
└──   plotting.py


##   🛠️ Requirements

* Python 3.7+
* pip (Python's package installer)

**Python   Libraries:**

* numpy
* pandas
* matplotlib
* statsmodels
* torch (PyTorch)
* scikit-learn
* networkx   (Optional: for causal graph visualization in Chapter 4)

##   📦 Installation

1.  **Clone   the repository:**

    ```bash
    git clone   [https://github.com/your-github-username/CausalDL-Code.git](https://github.com/your-github-username/CausalDL-Code.git)  # Replace with the correct URL
    cd   CausalDL-Code
    ```

2.  **Install   the required libraries:**

    ```bash
    pip install   numpy pandas matplotlib statsmodels torch scikit-learn  # Install core libraries
    pip install   networkx  # Install networkx if needed (for Chapter 4)
    ```

##   🚀 Usage

To   run the code examples:

1.  Navigate   to the directory of the chapter you're interested in.
2.  Execute   the Python script.

For   example:

```bash
cd chapter_1
python   correlation_vs_causation.py
⚠️ Disclaimer
These   code examples are provided for educational purposes to illustrate the   concepts discussed in the book. They are simplified for clarity and may   not be optimized for performance or suitable for production environments.   Always adapt and validate the code for your specific use case. Refer to   the book for detailed explanations, theoretical background, and best   practices.

🤝 Contributions
Contributions   to this repository are welcome! If you find a bug, have a suggestion,   or want to add an example, please:

Fork the repository.
Create a new branch for your changes.
Commit your changes with clear and descriptive commit messages.
Push your changes to your fork.
Submit a pull request.
Please   ensure your code adheres to basic Python style guidelines (PEP 8).

📄 License
This   code is released under the MIT   License. You are free to use,   modify, and distribute it for both commercial and non-commercial purposes.   See the LICENSE file for the full license text.

🔗 Related Resources
Book Website/Page: [Link to the book's website or publisher page]
Author's Website/Blog: [Link to your website or blog]
Online Forum/Discussion: [Link to a forum or discussion group for the book, if any]
Author
Durai   Rajamanickam

LinkedIn: [Your LinkedIn Profile]
Email: [Your Email Address (Optional)]
