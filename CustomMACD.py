import string
import numpy as np
import pandas as pd
from backtesting import VectorizedBacktester


class CustomMACD(VectorizedBacktester.VectorizedBacktester):
    def __init__(
        self,
        symbol: string,
        start: string,
        end: string,
        tc: float,
        ema_s: int = 12,
        ema_l: int = 26,
        sma_xl: int = 200,
        signal_smooth: int = 9,
        RSI_window=14,
        buy_thresh=70,
        short_thresh=30,
        granularity="1d"
    ):
        self.ema_s = ema_s
        self.ema_l = ema_l
        self.sma_xl = sma_xl
        self.signal_smooth = signal_smooth
        self.RSI_window = RSI_window
        self.buy_thresh = buy_thresh
        self.short_thresh = short_thresh
        super().__init__(symbol, start, end, tc, granularity=granularity)

    def test_strategy(self):

        # MACD CALCULATIONS

        data = self._data.copy().dropna()

        data["log_returns"] = np.log(data.price / data.price.shift(1))

        data["EMA_S"] = data.price.ewm(span=self.ema_s, adjust=False).mean()
        data["EMA_L"] = data.price.ewm(span=self.ema_l, adjust=False).mean()

        data["MACD"] = data.EMA_S - data.EMA_L
        data["signal"] = data.MACD.ewm(span=self.signal_smooth, adjust=False).mean()
        # data["MACD_VOL"] = data.MACD - data.signal

        # RSI CALCULATIONS

        data["change"] = data.price.diff()
        data["gain"] = data.change.mask(data.change < 0, 0.0)
        data["loss"] = -data.change.mask(data.change > 0, -0.0)

        data["avg_gain"] = data.gain.rolling(self.RSI_window).mean()
        data["avg_loss"] = data.loss.rolling(self.RSI_window).mean()

        data["rsi"] = 100 - (100 / (1 + (data.avg_gain / data.avg_loss)))

        # SMA CALCULATION
        data["SMA_XL"] = data.price.rolling(self.sma_xl).mean()


        # SET POSITION 

        data["position"] = np.where((data.MACD > data.signal) & (data.price > data.SMA_XL) & (data.rsi > self.buy_thresh), 1, np.nan)
        data["position"] = np.where((data.MACD < data.signal) & (data.price < data.SMA_XL) & (data.rsi < self.short_thresh), -1, data["position"])

        #data["position"] = np.where((data.MACD > data.signal) & (data.rsi > self.buy_thresh), 1, np.nan)
        #data["position"] = np.where((data.MACD < data.signal) & (data.rsi < self.short_thresh), -1, data["position"])

        data.position = data.position.ffill().fillna(0)

        data["strategy"] = data["position"].shift(1) * data["log_returns"]

        data["trades"] = data.position.diff().fillna(0).abs()

        data["hits"] = np.sign(data.log_returns) * np.sign(data.position)

        data["strategy"] = data.strategy - data.trades * self.tc

        data.dropna(inplace=True)

        data["creturns"] = data["log_returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data

        perf = data["cstrategy"].iloc[-1]  # absolute performance of the strategy
        outperf = perf - data["creturns"].iloc[-1]  # out-/underperformance of strategy

        return round(perf, 6), round(outperf, 6)
