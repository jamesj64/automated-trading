import string
import numpy as np
from strategies import ForexTrader


class MACD(ForexTrader):
    def __init__(
        self,
        conf_file: string,
        instrument: string,
        bar_length: string,
        units: int,
        duration: int,
        EMA_S=20,
        EMA_L=50,
        signal_smooth=10,
    ):
        self.EMA_S = EMA_S
        self.EMA_L = EMA_L
        self.signal_smooth = signal_smooth
        super().__init__(conf_file, instrument, bar_length, units, duration)

    def define_strategy(self):
        df = self.raw_data.copy()

        df["EMA_S"] = df[self.instrument].ewm(span=self.EMA_S, adjust=False).mean()
        df["EMA_L"] = df[self.instrument].ewm(span=self.EMA_L, adjust=False).mean()
        df["MACD"] = df.EMA_S - df.EMA_L
        df["signal"] = df.MACD.ewm(span=self.signal_smooth, adjust=False).mean()
        df["position"] = np.where(df["MACD"] > df["signal"], 1, -1)

        self.data = df.copy()
