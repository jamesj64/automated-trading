import string
import numpy as np
import matplotlib.pyplot as plt

from util import Instrument

Instrument = Instrument.Instrument

plt.style.use("seaborn-v0_8")


class VectorizedBacktester:
    """Class for the vectorized backtesting of trading strategies."""

    def __init__(self, symbol: string, start: string, end: string, tc: float):
        """
        Parameters
        ----------
        symbol: str
            ticker symbol (instrument) to be backtested
        start: str
            start date for data import
        end: str
            end date for data import
        tc: float
            proportional transaction/trading costs per trade
        """
        self.results_overview = None
        self.tc = tc
        self.results = None
        self._instrument = Instrument(symbol, start, end)
        self._data = self._instrument.get_data()

    @classmethod
    def from_instrument(cls, instrument, tc):
        return cls(
            instrument.get_ticker(), instrument.get_start(), instrument.get_end(), tc
        )

    def __repr__(self):
        return "VectorizedBacktester(symbol={}, start={}, end={})".format(
            self._instrument.get_ticker(),
            self._instrument.get_start(),
            self._instrument.get_end(),
        )

    def test_strategy(self):
        """Backtests the simple Contrarian trading strategy. This should be overridden as it is strategy-specific."""

        data = self._data.copy().dropna()
        data["log_returns"] = np.log(data.price / data.price.shift(1))
        data["position"] = np.sign(data["log_returns"].rolling(1).mean()).mul(-1)
        data["strategy"] = data["position"].shift(1) * data["log_returns"]
        data.dropna(inplace=True)

        # determine the number of trades in each bar
        data["trades"] = data.position.diff().fillna(0).abs()

        # subtract transaction/trading costs from pre-cost return
        data.strategy = data.strategy - data.trades * self.tc

        data["creturns"] = data["log_returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data

        perf = data["cstrategy"].iloc[-1]  # absolute performance of the strategy
        outperf = perf - data["creturns"].iloc[-1]  # out-/underperformance of strategy

        return round(perf, 6), round(outperf, 6)

    def plot_results(self):
        """Plots the performance of the trading strategy and compares to "buy and hold"."""
        if self.results is None:
            print("Run test_strategy() first.")
        else:
            title = "{} | TC = {}".format(self._instrument.get_ticker(), self.tc)
            self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8))
