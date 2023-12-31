import string
import numpy as np
import matplotlib.pyplot as plt

from util import Instrument

Instrument = Instrument.Instrument

plt.style.use("seaborn-v0_8")


class VectorizedBacktester:
    """Class for the vectorized backtesting of trading strategies."""

    def __init__(self, symbol: string, start: string, end: string, tc: float, granularity: string="1d", source_file=None, trading_hour_range=(0, 23)):
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
        granularity: str
            bar length (1d is default)
        source_file: str
            path to csv file to use instead of yf
        trading_hour_range: (int, int)
            range of hours to include (in New York time) in trading. 
        """
        self.trading_hour_range = trading_hour_range
        self.results_overview = None
        self.tc = tc
        self.results = None
        self._instrument = Instrument(symbol, start, end, source_file=source_file, granularity=granularity)
        self._data = self._instrument.get_data()

    @classmethod
    def from_instrument(cls, instrument, tc):
        instance =  cls(
            symbol=instrument.get_ticker(), start=instrument.get_start(), end=instrument.get_end(), tc=tc, granularity=instrument.granularity, source_file=instrument.source_file
        )
        instance._instrument = instrument
        instance._data = instance._instrument.get_data()
        return instance

    def __repr__(self):
        return "VectorizedBacktester(symbol={}, start={}, end={})".format(
            self._instrument.get_ticker(),
            self._instrument.get_start(),
            self._instrument.get_end(),
        )

    def test_strategy(self):
        """Backtests the simple Contrarian trading strategy. This should be overridden as it is strategy-specific."""

        (ts, te) = self.trading_hour_range

        data = self._data.copy().dropna()
        data["log_returns"] = np.log(data.price / data.price.shift(1))
        data["position"] = 1
        data["position"] = np.where((data.index.hour >= ts) & (data.index.hour <= te), data.position, 0)
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

    def hit_ratio(self):
        """Returns proporition of trades that are profitable"""
        if self.results is not None and self.results.hits is not None:
            value_count = self.results.hits.value_counts()
            return value_count[1] / (value_count[0] + value_count[-1] + value_count[1])
        return "No hit ratio avaliable. Test strategy before calling this method."
    
    def detailed_metrics(self):
        """Returns detailed performance metrics"""
        if self.results is not None:
            #value_count = self.results.hits.value_counts()
            #print("Hit Ratio: {}".format(value_count[1] / (value_count[0] + value_count[-1] + value_count[1])))
            
            mean_return = round(self.results.strategy.mean() * 252, 3)
            risk = round(self.results.strategy.std() * np.sqrt(252), 3)
            print("Annualized Return: {} | Annualized Risk: {}".format(mean_return, risk))

            cagr = (self.results.cstrategy.iloc[-1] / self.results.cstrategy.iloc[0]) ** (365 / (self.results.index[-1] - self.results.index[0]).days) - 1
            print("CAGR: {}".format(cagr))

            sharpe = (cagr - 0.039) / risk
            print("SHARPE: {}".format(sharpe))
        else:
            return "No data avaliable. Test strategy before calling detailed_metrics()"