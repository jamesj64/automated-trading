import pandas as pd
import numpy as np

def returns(price):
    return np.log(price / price.shift(1))

def dir(returns):
    return np.where(returns > 0, 1, 0)

def sma_crossover(price, window_s=50, window_l=200):
    return price.rolling(window_s).mean() - price.rolling(window_l).mean()

def mean_reversion(price, window=50):
    return (price - price.rolling(window).mean()) / price.rolling(window).std()

def min(price, window=50):
    return price.rolling(window).min() / price - 1

def max(price, window=50):
    return price.rolling(window).max() / price - 1

def momentum(returns, window=3):
    return returns.rolling(window).mean()

def volume(returns, window=50):
    return returns.rolling(window).std()

def macd(price, ema_s=12, ema_l=26, signal_smooth=9):
    macd = price.ewm(span=ema_s, adjust=False).mean() - price.ewm(span=ema_l, adjust=False).mean()
    return macd - macd.ewm(span=signal_smooth, adjust=False).mean()

def rsi(price, window=14):
    change = price.diff()
    avg_gain = change.mask(change < 0, 0.0).rolling(window).mean()
    avg_loss = -change.mask(change > 0, -0.0).rolling(window).mean()
    return 100 - (100 / (1 + (avg_gain / avg_loss)))