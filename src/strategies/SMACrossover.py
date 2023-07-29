import string
import numpy as np
from strategies import ForexTrader


class SMACrossover(ForexTrader.ForexTrader):
    def __init__(
        self,
        conf_file: string,
        instrument: string,
        bar_length: string,
        units: int,
        duration: int,
        SMA_S=50,
        SMA_L=200,
    ):
        self.SMA_S = SMA_S
        self.SMA_L = SMA_L
        super().__init__(conf_file, instrument, bar_length, units, duration)

    def define_strategy(self):
        df = self.raw_data.copy()

        df["log_ret"] = np.log(df[self.instrument].div(df[self.instrument].shift(1)))
        df["SMA_S"] = df[self.instrument].rolling(self.SMA_S).mean()
        df["SMA_L"] = df[self.instrument].rolling(self.SMA_L).mean()
        df["position"] = np.where(df["SMA_S"] > df["SMA_L"], 1, -1)

        self.data = df.copy()
