import numpy as np

from scipy.stats.norm import cdf


class OptionsPricer():
    def __init(self, hist_data)__:
        self.hist_data = hist_data
        self._sigma = np.sqrt(252) * self.hist_data[-252:].std()
        
    def bachelier_thorp(self, K, T, option='call', S=self.hist_data[-1], r=0.02):
        # K: strike price
        # T: time to maturity (days)
        # option: either 'call' or 'put'
        # S: current price
        # r: interest rate
        # sigma: volatility of underlying asset
        
        d1 = (np.log(S / K) + (r + 0.5 * self._sigma ** 2) * T) / (self._sigma * np.sqrt(T))
        d2 = (np.log(S / K) + (r - 0.5 * self._sigma ** 2) * T) / (self._sigma * np.sqrt(T))
        
        if option == 'call':
            price = (S * cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
        elif option == 'put':
            price = (K * np.exp(-r * T) * cdf(-d2, 0.0, 1.0) - S * cdf(-d1, 0.0, 1.0))
            
        return price
        
        
    def black_scholes(self, K, T, option='call', S=self.hist_data[-1], r=0.02):
        # K: strike price
        # T: time to maturity (days)
        # option: either 'call' or 'put'
        # S: current price
        # r: interest rate
        # sigma: volatility of underlying asset

        return bachelier_thorp(K, T, option=option, S=S, r=r)