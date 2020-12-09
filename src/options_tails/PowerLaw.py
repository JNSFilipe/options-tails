import numpy as np

from scipy.special import zeta
from scipy.stats import ks_2samp
from scipy.optimize import minimize_scalar, brute

class PowerLaw():
    def __init__(self):
        self._alpha = np.nan
        self._xmin   = np.nan
        

    def fit(self, x):
        ## Sort Values of x
        x = np.sort(x)
        
        ## Based on "POWER-LAW DISTRIBUTIONS IN EMPIRICAL DATA"
        ## Aaron Clauset et al
        n = len(x)
        
        ## Estimate xmin by maximising the 2 sample Kolmogorov-Smirnov test
        def ks(i, x):
            i = int(i)
            return ks_2samp(x[:i], x[i:])[0]
        r = brute(lambda i: ks(i,x), ranges=(slice(1,n,1),), disp=True, finish=None)
        self._xmin = int(r)
        
        ## Estimate alpha by maximising the maximum likelihood function
        def L(a, x, x_min, n):
            l = -n*np.log(zeta(a,q=x[x_min]))-a*np.sum([np.log(x[i]) for i in range(x_min,n)])
            return -l
        self._alpha = minimize_scalar(lambda a: L(a, x, self._xmin, n)).x
