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

# Mini Case Study 5: Effect of Remote Work on Employee Productivity
# From "Causal Inference for Machine Learning Engineers - A Practical Guide"
# by Durai Rajamanickam

# --- Simulating remote work data ---

import numpy as np

np.random.seed(2026)
n = 1000
Discipline = np.random.normal(0, 1, size=n)

# Propensity to choose remote work
logit_p = 1.2 * Discipline
p = 1 / (1 + np.exp(-logit_p))
RemoteWork = np.random.binomial(1, p)

# Productivity outcome
Productivity_baseline = 50 + 10 * Discipline + np.random.normal(0, 5, n)
Treatment_effect = 3  # Remote work increases productivity by 3 units
Productivity = Productivity_baseline + RemoteWork * Treatment_effect

X = Discipline.reshape(-1, 1)

