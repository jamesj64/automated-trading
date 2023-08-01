import string
import numpy as np
from strategies import ForexTrader


class ModdedMACD(ForexTrader.ForexTrader):
    def __init__(
        self,
        conf_file: string,
        instrument: string,
        bar_length: string,
        units: int,
        duration: int,
        EMA_S=12,
        EMA_L=26,
        EMA_XL=200,
        signal_smooth=9,
    ):
        self.EMA_S = EMA_S
        self.EMA_L = EMA_L
        self.EMA_XL = EMA_XL
        self.signal_smooth = signal_smooth
        super().__init__(conf_file, instrument, bar_length, units, duration)

    def define_strategy(self):
        df = self.raw_data.copy()

        df["EMA_S"] = df[self.instrument].ewm(span=self.EMA_S, adjust=False).mean()
        df["EMA_L"] = df[self.instrument].ewm(span=self.EMA_L, adjust=False).mean()
        df["EMA_XL"] = df[self.instrument].ewm(span=self.EMA_XL, adjust=False).mean()

        df["MACD"] = df.EMA_S - df.EMA_L
        df["signal"] = df.MACD.ewm(span=self.signal_smooth, adjust=False).mean()

        df["position"] = np.where((df["MACD"] > df["signal"]) & (df[self.instrument] > df.EMA_XL), 1, np.nan)
        df["position"] = np.where((df["MACD"] < df["signal"]) & (df[self.instrument] < df.EMA_XL), -1, df.position)

        df.position = df.position.ffill().fillna(0)

        self.data = df.copy()
