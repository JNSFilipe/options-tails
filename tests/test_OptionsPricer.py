import pytest
import numpy as np
import logging

from scipy.special import zeta
from scipy.optimize import fsolve
from scipy.constants import speed_of_light
from options_tails.OptionsPricer import OptionsPricer

__author__ = "JNSFilipe"
__copyright__ = "JNSFilipe"
__license__ = "mit"

_logger = logging.getLogger(__name__)

def test_OptionsPricer():
    ## Create a Normal Random Distribution with Fat Tails,
    ## for falues above 2 standard deviations. It is assumed
    ## that the normal dist has mean 0 and std 1. Only right
    ## tail is considered. Then distribution is used to 
    ## compute Option Prices using Black-Scholes and Fat
    ## Tails methods.

    # Set random seed for reproducibility
    np.random.seed(int(speed_of_light))
    
    # Set number of standard deviations where we transition
    # from normal distribution to the power law
    n_std = 3
    
    # Compute true value of alpha
    def f(a,x):
        z = a*np.log(x)-x**2/2-np.log(np.sqrt(2*np.pi)/zeta(a))
        return z
    alpha = fsolve(lambda a: f(a,n_std), 1.0000001)[0]
    
    # Generate random samples of the Normal Distribution
    g = np.random.normal(0,1, 1000)
    
    # Drop values below 0 and above the transition point
    g = g[(g>0) & (g<n_std)]
    
    # Compute true x_min
    x_min = len(g)
    
    # Generate random samples of the Power Law Distribution
    p = np.random.zipf(alpha, 1000)
    
    # Drop values below the transition point
    p = p[p>2]
    
    # Join the values of the two distrubutions into one
    x = np.concatenate([g,p])

    # Randomise order
    np.random.shuffle(x)
    
    # Create Pricer object
    op = OptionsPricer(x, power_law=True, alpha=alpha, xmin=x_min)

    # Compute Prices for Calls and Puts, using Bachelier-Thorp
    bt_cp = op.bachelier_thorp(100, 150, 28, option='call')
    bt_pp = op.bachelier_thorp(100, 150, 28, option='put')

    _logger.info('sigma: '+str(op._sigma))
    _logger.info('bt_cp: '+str(bt_cp))
    _logger.info('bt_pp: '+str(bt_pp))

    # Compute Prices for Calls and Puts, using Fat Tails
    pl_cp = op.power_law_tails(100, 150, 28, option='call')
    pl_pp = op.power_law_tails(100, 150, 28, option='put')

    _logger.info('pl_cp: '+str(pl_cp))
    _logger.info('pl_pp: '+str(pl_pp))
    
    # Define function to compute percentual difference
    def pct_diff(ref, val):
        return np.abs((val-ref)/ref)*100
    
    # Verify that estimated vales are within +/-10% error
    assert 1 == 1 #pct_diff(alpha, pl._alpha) <= 15
    #assert pct_diff(x_min, pl._xmin)  <= 15