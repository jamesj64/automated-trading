import string
import numpy as np
from strategies import ForexTrader

class Bollinger(ForexTrader.ForexTrader):
    def __init__(self, conf_file: string, instrument: string, bar_length: string, units: int, duration: int, window=5, dev=1):
        self.window = window
        self.dev = dev
        super().__init__(conf_file, instrument, bar_length, units, duration)

    def define_strategy(self):
        df = self.raw_data.copy()

        df["SMA"] = df[self.instrument].rolling(self.window).mean()
        df["Upper"] = df.SMA + self.dev * df[self.instrument].rolling(self.window).std()
        df["Lower"] = df.SMA - self.dev * df[self.instrument].rolling(self.window).std()
        df["distance"] = df[self.instrument] - df.SMA

        df["position"] = np.where(df[self.instrument] > df.Upper, -1, np.nan)
        df["position"] = np.where(df[self.instrument] < df.Lower, 1, df["position"])
        df["position"] = np.where(df.distance * df.distance.shift(1) < 0, 0, df["position"])
        df.position = df.position.ffill().fillna(0)

        self.data = df.copy()