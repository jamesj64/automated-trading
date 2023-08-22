import string
import numpy as np
import pandas as pd
from backtesting.VectorizedBacktester import VectorizedBacktester


class ADX(VectorizedBacktester):
    def __init__(
        self,
        symbol: string,
        start: string,
        end: string,
        tc: float,
        granularity="1d",
        window=14  
    ):
        self.window = window
        super().__init__(symbol, start, end, tc, granularity=granularity)

    def test_strategy(self):
        data = self._data.copy().dropna()

        data["log_returns"] = np.log(data.price / data.price.shift(1))

        data["high"] = data.price.rolling(self.window).max()

        data["low"] = data.price.rolling(self.window).min()

        data["pdm"] = data.high - data.high.shift(1)

        data["ndm"] = data.low.shift(1) - data.low

        data["tr"] = np.maximum(data.high - data.low, np.abs(data.high - data.price.shift(1)))

        data["tr"] = np.maximum(data.tr, np.abs(data.low - data.price.shift(1)))

        data["s_tr"] = data.tr.rolling(self.window).mean()

        data["s_pdm"] = data.pdm.rolling(self.window).mean()

        data["s_ndm"] = data.ndm.rolling(self.window).mean()

        data["pdi"] = (data["s_pdm"] / data["s_tr"]) * 100

        data["ndi"] = (data["s_ndm"] / data["s_tr"]) * 100

        data["dx"] = np.abs((data.pdi - data.ndi) / (data.pdi + data.ndi))

        data["adx"] = data.dx.rolling(self.window).mean()

        data["position"] = np.where((data.adx > 25) & (data.pdi > data.ndi), 1, np.nan)
        data["position"] = np.where((data.adx > 25) & (data.pdi < data.ndi), -1, data.position)

        data["position"] = data.position.ffill().fillna(0)

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