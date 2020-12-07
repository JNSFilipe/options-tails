import numpy as np

class PowerLaw():
    def __init__(self):
        self._alpha = np.nan
        seff_xmin   = np.nan

    def fit(self, x):
        # Based on "POWER-LAW DISTRIBUTIONS IN EMPIRICAL DATA", Aaron Clauset et al

