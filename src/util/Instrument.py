import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf


class Instrument:
    def __init__(self, ticker, start, end):
        self._ticker = ticker
        self._start = start
        self._end = end
        self._data = None
        self.get_data()
        self.log_returns()

    # PROPERTIES START
    def __repr__(self):
        return "Instrument(ticker={}, start={}, end={})".format(
            self._ticker, self._start, self._end
        )

    def get_start(self):
        return self._start

    def set_start(self, start):
        self._start = start
        self.get_data()

    def get_end(self):
        return self._end

    def set_end(self, end):
        self._end = end
        self.get_data()

    def get_ticker(self):
        return self._ticker

    def set_ticker(self, ticker):
        self._ticker = ticker
        self.get_data()

    # PROPERTIES END
    def get_data(self):
        data = yf.download(self._ticker, self._start, self._end).Close.to_frame()
        data.rename(columns={"Close": "price"}, inplace=True)
        self._data = data
        return self._data.copy()

    def log_returns(self):
        self._data["log_returns"] = np.log(self._data.price / self._data.price.shift(1))

    def plot_prices(self):
        self._data.price.plot(figsize=(12, 8))
        plt.title("Price Chart: {}".format(self._ticker), fontsize=13)

    def plot_returns(self, kind="ts"):
        if kind == "ts":
            self._data.log_returns.plot(figsize=(12, 8))
            plt.title("Returns: {}".format(self._ticker), fontsize=15)
        elif kind == "hist":
            self._data.log_returns.hist(
                figsize=(12, 8), bins=int(np.sqrt(len(self._data)))
            )
            plt.title("Frequency of Returns: {}".format(self._ticker), fontsize=15)

    def mean_return(self, freq=None):
        if freq is None:
            return self._data.log_returns.mean()
        else:
            resampled_price = self._data.price.resample(freq).last()
            resampled_returns = np.log(resampled_price / resampled_price.shift(1))
            return resampled_returns.mean()

    def std_returns(self, freq=None):
        if freq is None:
            return self._data.log_returns.std()
        else:
            resampled_price = self._data.price.resample(freq).last()
            resampled_returns = np.log(resampled_price / resampled_price.shift(1))
            return resampled_returns.std()

    def annualized_performance(self):
        mean_return = round(self._data.log_returns.mean() * 252, 3)
        risk = round(self._data.log_returns.std() * np.sqrt(252), 3)
        print("Return: {} | Risk: {}".format(mean_return, risk))
