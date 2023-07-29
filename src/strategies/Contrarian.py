import string
import numpy as np
from strategies import ForexTrader

class Contrarian(ForexTrader.ForexTrader):

    def __init__(self, conf_file: string, instrument: string, bar_length: string, units: int, duration: int, window=1):
        self.window = window
        super().__init__(conf_file, instrument, bar_length, units, duration)

    def define_strategy(self):
        df = self.raw_data.copy()

        df["log_ret"] = np.log(df[self.instrument].div(df[self.instrument].shift(1)))
        df["position"] = -np.sign(df.log_ret.rolling(self.window).mean())

        self.data = df.copy()