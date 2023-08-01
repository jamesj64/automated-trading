import string
import numpy as np
from strategies import ForexTrader


class MACDRSI(ForexTrader.ForexTrader):
    def __init__(
        self,
        conf_file: string,
        instrument: string,
        bar_length: string,
        units: int,
        duration: int,
        EMA_S=12,
        EMA_L=26,
        signal_smooth=9,
        RSI_window=14,
        buy_thresh=70,
        short_thresh=30
    ):
        self.EMA_S = EMA_S
        self.EMA_L = EMA_L
        self.signal_smooth = signal_smooth
        self.RSI_window = RSI_window
        self.buy_thresh=buy_thresh
        self.short_thresh=short_thresh
        super().__init__(conf_file, instrument, bar_length, units, duration)

    def define_strategy(self):
        df = self.raw_data.copy()

        # RSI CALCULATIONS

        df["change"] = df[self.instrument].diff()
        df["gain"] = df.change.mask(df.change < 0, 0.0)
        df["loss"] = -df.change.mask(df.change > 0, -0.0)

        df["avg_gain"] = df.gain.rolling(self.window).mean()
        df["avg_loss"] = df.loss.rolling(self.window).mean()

        df["rsi"] = 100 - (100 / (1 + (df.avg_gain / df.avg_loss)))

        # MACD CALCULATIONS

        df["EMA_S"] = df[self.instrument].ewm(span=self.EMA_S, adjust=False).mean()
        df["EMA_L"] = df[self.instrument].ewm(span=self.EMA_L, adjust=False).mean()
        df["MACD"] = df.EMA_S - df.EMA_L
        df["signal"] = df.MACD.ewm(span=self.signal_smooth, adjust=False).mean()

        # GO LONG IF MACD > SIGNAL AND RSI > buy_thresh
        # GO SHORT IF MACD < SIGNAL AND RSI < sell_thresh
        # OTHERWISE MAINTAIN CURRENT POSITION

        df["position"] = np.where((df.rsi > self.buy_thresh) & (df.MACD > df.signal), 1, np.nan)
        df["position"] = np.where((df.rsi < self.sell_thresh) & (df.MACD < df.signal), -1, df.position)
        df.position = df.position.ffill().fillna(0)

        self.data = df.copy()
