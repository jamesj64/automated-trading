import string
import numpy as np
from strategies import ForexTrader


class IchimokuCloud(ForexTrader.ForexTrader):
    def __init__(
        self,
        conf_file: string,
        instrument: string,
        bar_length: string,
        units: int,
        duration: int,
        conversion=9,
        base=26,
        leading_l=52,
    ):
        self.conversion = conversion
        self.base = base
        self.leading_l = leading_l
        super().__init__(conf_file, instrument, bar_length, units, duration)

    def define_strategy(self):
        df = self.raw_data.copy()

        df["conversion"] = (df[self.instrument].rolling(self.conversion).max() + df[self.instrument].rolling(self.conversion).min()) / 2

        df["base"] = (df[self.instrument].rolling(self.base).max() + df[self.instrument].rolling(self.base).min()) / 2

        df["leading_s"] = (df.conversion + df.base) / 2

        df["leading_l"] = (df[self.instrument].rolling(self.leading_l).max() + df[self.instrument].rolling(self.leading_l).min()) / 2

        # When price is above cloud and leading_s is above leading_l, go long
        # When price is below cloud and leading_s is below leading_l, go short
        # Otherwise, maintain current position
        df["position"] = np.where((df[self.instrument] > df["leading_s"]) & (df["leading_s"] > df["leading_l"]), 1, np.nan)
        df["position"] = np.where((df[self.instrument] < df["leading_s"]) & (df["leading_s"] < df["leading_l"]), -1, df.position)

        df.position = df.position.ffill().fillna(0)

        self.data = df.copy()
