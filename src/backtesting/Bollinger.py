import string
import numpy as np
import pandas as pd
from backtesting.VectorizedBacktester import VectorizedBacktester

class Bollinger(VectorizedBacktester):
    def __init__(
        self,
        symbol: string,
        start: string,
        end: string,
        tc: float,
        window:int = 5,
        dev:int = 1,
        granularity="1d",
        source_file=None,
        trading_hour_range=(0, 23)
    ):
        self.window = window
        self.dev = dev
        self.trading_hour_range=trading_hour_range
        super().__init__(symbol, start, end, tc, granularity=granularity, source_file=source_file)

    def test_strategy(self):

        (ts, te) = self.trading_hour_range

        data = self._data.copy().dropna()

        data["log_returns"] = np.log(data.price / data.price.shift())

        data["sma"] = data.price.rolling(self.window).mean()
        data["upper"] = data.sma + self.dev * data.price.rolling(self.window).std()
        data["lower"] = data.sma - self.dev * data.price.rolling(self.window).std()

        data["distance"] = data.price - data.sma

        data["position"] = np.where(data.price > data.upper, -1, np.nan)
        data["position"] = np.where(data.price < data.lower, 1, data["position"])
        data["position"] = np.where(
            data.distance * data.distance.shift(1) < 0, 0, data["position"]
        )

        data["position"] = np.where((data.index.hour >= ts) & (data.index.hour <= te), data.position, np.nan)

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