import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util import Instrument

Instrument = Instrument.Instrument

plt.style.use("seaborn-v0_8")


class IterativeBacktester:
    """Base class for iterative (event-driven) backtesting of trading strategies."""

    def __init__(
        self,
        symbol: string,
        start: string,
        end: string,
        amount: int,
        use_spread=False,
        source_file=None,
    ):
        """
        Parameters
        ----------
        symbol: str
            ticker symbol (instrument) to be backtested
        start: str
            start date for data import
        end: str
            end date for data import
        amount: float
            initial amount to be invested per trade
        use_spread: boolean (default = True)
            whether trading costs (bid-ask spread) are included
        source_file: string (default = None)
            source file to read from. necessary for use_spread to be enabled.
        """
        self.symbol = symbol
        self.start = start
        self.end = end
        self.initial_balance = amount
        self.current_balance = amount
        self.units = 0
        self.trades = 0
        self.position = 0
        self.use_spread = (
            use_spread and source_file is not None
        )  # Can't use_spread if no source file bc yf no spread
        self.data = None
        self._instrument = Instrument(symbol, start, end, source_file)
        self.get_data()

    @classmethod
    def from_instrument(cls, instrument: Instrument, amount, use_spread=True):
        return cls(
            instrument.get_ticker(),
            instrument.get_start(),
            instrument.get_end(),
            amount,
            use_spread,
            source_file=instrument.source_file,
        )

    def get_data(self):
        """Gets data from instrument"""
        raw = self._instrument.get_data()
        raw["returns"] = np.log(raw.price / raw.price.shift(1))
        self.data = raw

    def plot_data(self, cols=None):
        """Plots the closing price for the symbol."""
        if cols is None:
            cols = "price"
        self.data[cols].plot(figsize=(12, 8), title=self.symbol)

    def get_values(self, bar):
        """Returns the date, the price and the spread for the given bar."""
        date = str(self.data.index[bar].date())
        price = round(self.data.price.iloc[bar], 5)
        spread = None if not self.use_spread else round(self.data.spread.iloc[bar], 5)
        return date, price, spread

    def print_current_balance(self, bar):
        """Prints out the current (cash) balance."""
        date, price, spread = self.get_values(bar)
        print("{} | Current Balance: {}".format(date, round(self.current_balance, 2)))

    def buy_instrument(self, bar, units=None, amount=None):
        """Places and executes a buy order (market order)."""
        date, price, spread = self.get_values(bar)
        if self.use_spread:
            price += spread / 2  # ask price
        if (
            amount is not None
        ):  # use units if units are passed, otherwise calculate units
            units = int(amount / price)
        self.current_balance -= units * price  # reduce cash balance by "purchase price"
        self.units += units
        self.trades += 1
        print("{} |  Buying {} for {}".format(date, units, round(price, 5)))

    def sell_instrument(self, bar, units=None, amount=None):
        """Places and executes a sell order (market order)."""
        date, price, spread = self.get_values(bar)
        if self.use_spread:
            price -= spread / 2  # bid price
        if (
            amount is not None
        ):  # use units if units are passed, otherwise calculate units
            units = int(amount / price)
        self.current_balance += (
            units * price
        )  # increases cash balance by "purchase price"
        self.units -= units
        self.trades += 1
        print("{} |  Selling {} for {}".format(date, units, round(price, 5)))

    def go_long(self, bar, units=None, amount=None):
        if self.position == -1:
            self.buy_instrument(
                bar, units=-self.units
            )  # if short position, go neutral first
        if units:
            self.buy_instrument(bar, units=units)
        elif amount:
            if amount == "all":
                amount = self.current_balance
            self.buy_instrument(bar, amount=amount)  # go long

    def go_short(self, bar, units=None, amount=None):
        if self.position == 1:
            self.sell_instrument(
                bar, units=self.units
            )  # if long position, go neutral first
        if units:
            self.sell_instrument(bar, units=units)
        elif amount:
            if amount == "all":
                amount = self.current_balance
            self.sell_instrument(bar, amount=amount)

    def print_current_position_value(self, bar):
        """Prints out the current position value."""
        date, price, spread = self.get_values(bar)
        cpv = self.units * price
        print("{} |  Current Position Value = {}".format(date, round(cpv, 2)))

    def print_current_nav(self, bar):
        """Prints out the current net asset value (nav)."""
        date, price, spread = self.get_values(bar)
        nav = self.current_balance + self.units * price
        print("{} |  Net Asset Value = {}".format(date, round(nav, 2)))

    def test_strategy(self):
        pass

    def close_pos(self, bar):
        """Closes out a long or short position (go neutral)."""
        date, price, spread = self.get_values(bar)
        print(75 * "-")
        print("{} | +++ CLOSING FINAL POSITION +++".format(date))
        self.current_balance += self.units * price  # closing final position
        if self.use_spread:
            self.current_balance -= (
                abs(self.units) * spread / 2
            )  # subtract half-spread costs
        print("{} | closing position of {} for {}".format(date, self.units, price))
        self.units = 0  # setting position to neutral
        self.trades += 1
        perf = (
            (self.current_balance - self.initial_balance) / self.initial_balance * 100
        )
        self.print_current_balance(bar)
        print("{} | net performance (%) = {}".format(date, round(perf, 2)))
        print("{} | number of trades executed = {}".format(date, self.trades))
        print(75 * "-")

    def reset(self):
        self.position = 0  # initial neutral position
        self.trades = 0  # no trades yet
        self.current_balance = self.initial_balance  # reset initial capital
        self.get_data()  # reset dataset
