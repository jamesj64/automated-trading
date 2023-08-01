import string
import numpy as np
import pandas as pd
from backtesting import VectorizedBacktester


class CustomMACD(VectorizedBacktester.VectorizedBacktester):
    def __init__(
        self,
        symbol: string,
        start: string,
        end: string,
        tc: float,
        ema_s: int = 12,
        ema_l: int = 26,
        ema_xl: int = 200,
        signal_smooth: int = 9
    ):
        self.ema_s = ema_s
        self.ema_l = ema_l
        self.ema_xl = ema_xl
        self.signal_smooth = signal_smooth
        super().__init__(symbol, start, end, tc)

    def test_strategy(self):

        data = self._data.copy().dropna()

        data["log_returns"] = np.log(data.price / data.price.shift(1))

        data["EMA_S"] = data.price.ewm(span=self.ema_s, adjust=False).mean()
        data["EMA_L"] = data.price.ewm(span=self.ema_l, adjust=False).mean()
        data["EMA_XL"] = data.price.ewm(span=self.ema_xl, adjust=False).mean()

        data["MACD"] = data.EMA_S - data.EMA_L
        data["signal"] = data.MACD.ewm(span=self.signal_smooth, adjust=False).mean()

        data["MACD_VOL"] = data.MACD - data.signal

        data["MACD_MA"] = data.MACD_VOL.ewm(span=20, adjust=False).mean()

        data["MACD_SIGNAL"] = np.sign(data.MACD - data.signal)

        data["position"] = np.where((data.MACD > data.signal) & ((data.MACD_MA - data.MACD_MA.shift(1) > 0)), 1, np.nan)
        data["position"] = np.where((data.MACD < data.signal) & ((data.MACD_MA - data.MACD_MA.shift(1) < 0)), -1, data["position"])

        data.position = data.position.ffill().fillna(0)

        data["trades"] = data.position.diff().fillna(0).abs()

        #data["position_changes"] = np.where((data.position != data.position.shift(1)) & (data.position.shift(1) != np.nan), 1, 0)

        #df = data.loc[data.position != data.position.shift(1)].price.to_frame()

        #print(np.sign(df - df.shift(1)).value_counts())

        data["hits"] = np.sign(data.log_returns) * np.sign(data.position)

        data["strategy"] = data.position.shift(1) * data.log_returns - data.trades * self.tc

        data.dropna(inplace=True)

        data["creturns"] = data["log_returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data

        perf = data["cstrategy"].iloc[-1]  # absolute performance of the strategy
        outperf = perf - data["creturns"].iloc[-1]  # out-/underperformance of strategy

        return round(perf, 6), round(outperf, 6)
