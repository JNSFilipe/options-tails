import numpy as np

from PowerLaw import PowerLaw
from scipy.stats.norm import cdf


class OptionsPricer():
    def __init__(self, hist_returns, power_law=False):
        self.hist_data = hist_returns
        self._sigma = np.sqrt(252) * self.hist_returns[-252:].std()
        
        if power_law:
            self._pl_right = PowerLaw()
            self._pl_left  = PowerLaw()
            
            self._pl_right.fit(self.hist_returns[self.hist_returns>0])
            self._pl_left.fit(self.hist_returns[self.hist_returns<0])
            
        
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
        
    def power_law_tails(self, K, T, option='call', S=self.hist_data[-1], anchor='black-scholes'):
        if S < K:
            alpha = self._pl_right._alpha
            xmin = self._pl_right._xmin
            K_anchor = S + np.sort(self.hist_returns[self.hist_returns>0])[xmin]
        elif S > K:
            alpha = self._pl_left._alpha
            xmin = self._pl_left._xmin
            K_anchor = S + np.sort(self.hist_returns[self.hist_returns<0])[xmin]
            
        if anchor == 'black-scholes' or anchor == 'bachelier-thorp':
            P_anchor = self.bachelier_thorp(K_anchor, T, option=option, S=S, r=r)
        if anchor == 'market':
            #TODO
            pass
            
        #l = ((alpha-1)*S*P_anchor**(1-alpha))**(1/alpha)
        P = (K/K_anchor)**(1-alpha)*P_anchor
        
        return P