import string
import numpy as np
from strategies import ForexTrader


class EMA(ForexTrader.ForexTrader):
    def __init__(
        self,
        conf_file: string,
        instrument: string,
        bar_length: string,
        units: int,
        duration: int,
        EMA_S=20,
        EMA_L=50,
    ):
        self.EMA_S = EMA_S
        self.EMA_L = EMA_L
        super().__init__(conf_file, instrument, bar_length, units, duration)

    def define_strategy(self):
        df = self.raw_data.copy()

        df["EMA_S"] = df[self.instrument].ewm(span=self.EMA_S, adjust=False).mean()
        df["EMA_L"] = df[self.instrument].ewm(span=self.EMA_L, adjust=False).mean()
        df["position"] = np.where(df["EMA_S"] > df["EMA_L"], 1, -1)

        self.data = df.copy()
