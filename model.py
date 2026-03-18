from __future__ import annotations
from typing import List
import numpy as np

# names of five input variables shown to user in calculator ui
FEATURES = ["Initial Structural Strength (MPa)", "Thickness (mm)", "Speed (m/s)", "Cooling Fans Power (W)", "Number of cooling fans"]

# name of output variable predicted by model
TARGET = "Final Structural Strength (MPa)"

# linear regression parameters
W = np.array([364.54435751864537, 2.7268522381452276, 16.68207764207219, 8.930510176877332, -4.91824389113619, -14.888434270235448], dtype=float)

# mean value of each input feature in training data
MU = np.array([594.0162443144899, 3.247568810916178, 4.30462508619557, 893.7914230019493, 48.512020792722545], dtype=float)

# standard deviation of each input feature in training data
SIGMA = np.array([15.915684963900695, 0.4786019232456688, 0.33270119546324906, 567.0908296976912, 5.339696959818941], dtype=float)

# allowed ui ranges for each input feature
RANGES = [[500, 700], [1, 6], [2, 7], [0, 2000], [0, 100]]

# y axis plotting limits for live prediction chart
Y_LIM = [200, 500]

# predicts final structural strength from raw 5 feature input vector
def predict(x_raw: List[float]) -> float:
    x = np.asarray(x_raw, dtype=float)
    xs = (x - MU) / SIGMA
    return float(W[0] + xs @ W[1:])
