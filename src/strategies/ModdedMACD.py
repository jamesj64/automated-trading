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
        SMA_XL=200,
        signal_smooth=9,
        RSI_window=14,
    ):
        self.EMA_S = EMA_S
        self.EMA_L = EMA_L
        self.SMA_XL = SMA_XL
        self.signal_smooth = signal_smooth
        self.RSI_window = RSI_window
        super().__init__(conf_file, instrument, bar_length, units, duration)

    def define_strategy(self):
        df = self.raw_data.copy()

        df["log_returns"] = np.log(df[self.instrument] / df[self.instrument].shift(1))

        df["EMA_S"] = df[self.instrument].ewm(span=self.EMA_S, adjust=False).mean()
        df["EMA_L"] = df[self.instrument].ewm(span=self.EMA_L, adjust=False).mean()
        df["SMA_XL"] = df[self.instrument].rolling(self.SMA_XL).mean()

        df["MACD"] = df.EMA_S - df.EMA_L
        df["signal"] = df.MACD.ewm(span=self.signal_smooth, adjust=False).mean()

        df["change"] = df[self.instrument].diff()
        df["gain"] = df.change.mask(df.change < 0, 0.0)
        df["loss"] = -df.change.mask(df.change > 0, -0.0)

        df["avg_gain"] = df.gain.rolling(self.RSI_window).mean()
        df["avg_loss"] = df.loss.rolling(self.RSI_window).mean()

        df["rsi"] = 100 - (100 / (1 + (df.avg_gain / df.avg_loss)))

        df["SMA_XL"] = df[self.instrument].rolling(self.SMA_XL).mean()

        df["position"] = np.where((df["MACD"] > df["signal"]) & (df[self.instrument] > df.SMA_XL) & (df["rsi"] > 70), 1, np.nan)
        df["position"] = np.where((df["MACD"] < df["signal"]) & (df[self.instrument] < df.SMA_XL) & (df["rsi"] < 30), -1, df.position)

        df.position = df.position.ffill().fillna(0)

        self.data = df.copy()
