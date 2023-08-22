import string
import keras
import pickle
import numpy as np
from util import Calculations as calc
from backtesting.VectorizedBacktester import VectorizedBacktester

class DNN(VectorizedBacktester):
    def __init__(
        self,
        symbol: string,
        start: string,
        end: string,
        tc: float,
        granularity="1d",
        source_file=None,
        model: string = None,
        pkl: string = None,
        lags=5,
    ):
        self.model = model
        self.pkl = pkl
        self.lags = lags
        self.load_model(model, pkl)
        super().__init__(symbol, start, end, tc, granularity=granularity, source_file=source_file)

    def load_model(self, model_path: string, pkl_path: string):
        self.model = keras.models.load_model(model_path)
        params = pickle.load(open(pkl_path, "rb"))
        self.mean = params["mean"]
        self.std = params["std"]
        
    def test_strategy(self):
        df = self._data.copy().dropna()

        df["returns"] = calc.returns(df.price)
        df["dir"] = calc.dir(df.returns)
        df["sma"] = calc.sma_crossover(df.price)
        df["mean_reversion"] = calc.mean_reversion(df.price)
        df["min"] = calc.min(df.price)
        df["max"] = calc.min(df.price)
        df["mom"] = calc.momentum(df.returns)
        df["vol"] = calc.volume(df.returns)
        df["macd"] = calc.macd(df.price)
        df["rsi"] = calc.rsi(df.price)
        df.dropna(inplace=True)

        cols = []
        features = [
            "returns",
            "dir",
            "sma",
            "mean_reversion",
            "min",
            "max",
            "mom",
            "vol",
            "macd",
            "rsi"
        ]

        for f in features:
            for lag in range(1, self.lags + 1):
                col = "{}_lag_{}".format(f, lag)
                df[col] = df[f].shift(lag)
                cols.append(col)
        df.dropna(inplace=True)

        df_s = (df - self.mean) / self.std

        df["prob"] = self.model.predict(df_s[cols])

        df["s_prob"] = df.prob.rolling(50).mean()

        df["position"] = np.where(df.s_prob < 0.498, -1, np.nan)
        df["position"] = np.where(df.s_prob > 0.51, 1, df.position)
        df["position"] = df.position.ffill().fillna(0)

        df["strategy"] = df["position"].shift(1) * df["returns"]

        df["trades"] = df.position.diff().fillna(0).abs()

        df["hits"] = np.sign(df.returns) * np.sign(df.position)

        df["strategy"] = df.strategy - df.trades * self.tc

        df.dropna(inplace=True)

        df["creturns"] = df["returns"].cumsum().apply(np.exp)
        df["cstrategy"] = df["strategy"].cumsum().apply(np.exp)
        self.results = df

        perf = df["cstrategy"].iloc[-1]  # absolute performance of the strategy
        outperf = perf - df["creturns"].iloc[-1]  # out-/underperformance of strategy

        return round(perf, 6), round(outperf, 6)