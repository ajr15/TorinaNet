import numpy as np
from scipy import integrate

BOLTZMANS_CONSTANT = 1.38e-23
JOULE_TO_EV = 6.242e18
HARTREE_TO_EV = 27.2114

class MaxwellBoltzmannDistribution:

    def __init__(self, T: float):
        self.T = T

    def density(self, energy: float):
        """Return the probability density for a given energy (in eV)"""
        x = energy / JOULE_TO_EV
        return 2 * np.sqrt(x / np.pi) * np.power(1 / (BOLTZMANS_CONSTANT * self.T), 1.5) * \
            np.exp(- x / (BOLTZMANS_CONSTANT * self.T))
    
    def cdf(self, energy: float):
        """Returns the commulative distribution function for a given energy (in eV)"""
        return integrate.quad(self.density, 0, energy)[0]
    
    def sf(self, energy: float):
        """Returns the survaval function for a given energy (probability of higher energy) energy in eV"""
        return integrate.quad(self.density, energy, np.inf)[0]
