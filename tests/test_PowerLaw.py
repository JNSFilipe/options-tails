import pytest
import numpy as np

from scipy.special import zeta
from scipy.optimize import fsolve
from options_tails.PowerLaw import PowerLaw

__author__ = "JNSFilipe"
__copyright__ = "JNSFilipe"
__license__ = "mit"


def test_PowerLaw():
    ## Create a Normal Random Distribution with Fat Tails,
    ## for falues above 2 standard deviations. It is assumed
    ## that the normal dist has mean 0 and std 1. Only right
    ## tail is considered. True Value of alpha must be prev.
    ## computed, by solvig the equation that equals the 
    ## normal distribution and the power law distribution.
    ## Lastly, we check if the error of the estimated alpha
    ## and x_min are within 10% of the true values
    
    # Set number of standard deviations where we transition
    # from normal distribution to the power law
    n_std = 2
    
    # Compute true value of alpha
    def f(a,x):
        z = a*np.log(x)-x**2/2-np.log(np.sqrt(2*np.pi)/zeta(a))
        return z
    alpha = fsolve(lambda a: f(a,n_std),1.0000001)[0]
    
    # Generate random samples of the Normal Distribution
    g = np.random.normal(0,1, 10000)
    
    # Drop values below 0 and above the transition point
    g = g[(g>0) & (g<n_std)]
    
    # Compute true x_min
    x_min = len(g)
    
    # Generate random samples of the Power Law Distribution
    p = np.random.zipf(r,10000)
    
    # Drop values below the transition point
    p = p[p>2]
    
    # Join the values of the two distrubutions into one
    x = np.concatenate([g,p])
    
    # Estimate power law parameters
    pl = PowerLaw
    pl.fit(x)
    
    # Define function to compute percentual difference
    def pct_diff(ref, val):
        return np.abs((val-ref)/ref)*100
    
    # Verify that estimated vales are within +/-10% error
    assert pct_diff(alpha, pl._alpha) <= 10
    assert pct_diff(x_min, pl._xmin)  <= 10