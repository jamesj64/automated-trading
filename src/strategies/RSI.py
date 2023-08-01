import string
import numpy as np
from strategies import ForexTrader


class RSI(ForexTrader.ForexTrader):
    def __init__(
        self,
        conf_file: string,
        instrument: string,
        bar_length: string,
        units: int,
        duration: int,
        window=14,
        buy_thresh=70,
        short_thresh=30
    ):
        self.window = window
        self.buy_thresh=buy_thresh
        self.short_thresh=short_thresh
        super().__init__(conf_file, instrument, bar_length, units, duration)

    def define_strategy(self):
        df = self.raw_data.copy()

        df["change"] = df[self.instrument].diff()
        df["gain"] = df.change.mask(df.change < 0, 0.0)
        df["loss"] = -df.change.mask(df.change > 0, -0.0)

        df["avg_gain"] = df.gain.rolling(self.window).mean()
        df["avg_loss"] = df.loss.rolling(self.window).mean()

        df["rsi"] = 100 - (100 / (1 + (df.avg_gain / df.avg_loss)))

        df["position"] = np.where(df.rsi > self.buy_thresh, 1, np.nan)
        df["position"] = np.where(df.rsi < df.short_thresh, -1, df.position)
        df.position = df.position.ffill().fillna(0)

        self.data = df.copy()
