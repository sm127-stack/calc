"""
Model wrapper extracted from your notebook.
Guest and member use the same predictor coefficients; member adds saving.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np

FEATURES = ["Initial Structural Strength (MPa)", "Thickness (mm)", "Speed (m/s)", "Cooling Fans Power (W)", "Number of cooling fans"]
TARGET = "Final Structural Strength (MPa)"
W = np.array([364.54435751864537, 2.7268522381452276, 16.68207764207219, 8.930510176877332, -4.91824389113619, -14.888434270235448], dtype=float)
MU = np.array([594.0162443144899, 3.247568810916178, 4.30462508619557, 893.7914230019493, 48.512020792722545], dtype=float)
SIGMA = np.array([15.915684963900695, 0.4786019232456688, 0.33270119546324906, 567.0908296976912, 5.339696959818941], dtype=float)

def predict(x_raw: List[float]) -> float:
    x = np.asarray(x_raw, dtype=float)
    xs = (x - MU) / SIGMA
    return float(W[0] + xs @ W[1:])
