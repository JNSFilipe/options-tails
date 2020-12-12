import yfinance as yf
import datetime as dt
import numpy as np

from options_tails.PowerLaw import PowerLaw
from scipy.stats import norm

__author__ = "JNSFilipe"
__copyright__ = "JNSFilipe"
__license__ = "mit"

class OptionsPricer():
    def __init__(self, hist_returns, power_law=False, vol_lookback_len=252, ticker=None):
        self.hist_returns = hist_returns
        self._sigma = np.sqrt(252) * self.hist_returns[-vol_lookback_len:].std()

        if not ticker is None:
            self.ticker = ticker
        
        if power_law:
            self._pl_right = PowerLaw()
            self._pl_left  = PowerLaw()
            
            self._pl_right.fit(self.hist_returns[self.hist_returns>0])
            self._pl_left.fit(self.hist_returns[self.hist_returns<0])
            
        
    def bachelier_thorp(self, K, T, option='call', S=None, r=0.02):
        # K: strike price
        # T: time to maturity (days)
        # option: either 'call' or 'put'
        # S: current price
        # r: interest rate
        # sigma: volatility of underlying asset

        if S is None:
            S = self.hist_returns[-1]
        
        d1 = (np.log(S / K) + (r + 0.5 * self._sigma ** 2) * T) / (self._sigma * np.sqrt(T))
        d2 = (np.log(S / K) + (r - 0.5 * self._sigma ** 2) * T) / (self._sigma * np.sqrt(T))
        
        if option == 'call':
            price = (S * norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * norm.cdf(d2, 0.0, 1.0))
        elif option == 'put':
            price = (K * np.exp(-r * T) * norm.cdf(-d2, 0.0, 1.0) - S * norm.cdf(-d1, 0.0, 1.0))
            
        return price
        
        
    def black_scholes(self, K, T, option='call', S=None, r=0.02):
        # K: strike price
        # T: time to maturity (days)
        # option: either 'call' or 'put'
        # S: current price
        # r: interest rate
        # sigma: volatility of underlying asset

        return self.bachelier_thorp(K, T, option=option, S=S, r=r)
        
    def power_law_tails(self, K, T, option='call', S=None, anchor='black-scholes', r=0.02, curr_date=dt.datetime.today().strftime('%Y-%m-%d')):
        # r: interest rate, only used if anchor=='black-scholes' or anchor=='bachelier-thorp'
        # curr_date: current date, only used if anchor=='market'
        if S is None:
            S = self.hist_returns[-1]
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
            # Get ticker handler
            stk = yf.Ticker(self.ticker)

            # Compute strike date
            stk_date = dt.datetime.strptime(curr_date, '%Y-%m-%d') + dt.timedelta(days=T)
            stk_date = stk_date.strftime('%Y-%m-%d')

            # Get actual option chain
            opt = stk.option_chain(stk_date)

            # Chose between calls and puts
            if option == 'calls':
                opt = opt.calls
            elif options == 'puts':
                opt = opt.puts

            #TODO
            # instead of hoping that the exact strike exists, compute nearest strike to get P_anchor and adjust K_anchor accordingly
            P_anchor = opt[opt['strike']==K_anchor].lastPrice.values[0]
            
        #l = ((alpha-1)*S*P_anchor**(1-alpha))**(1/alpha)
        P = (K/K_anchor)**(1-alpha)*P_anchor
        
        return P
