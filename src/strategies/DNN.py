import string
import keras
import pickle
import numpy as np
from util import Calculations as calc
from strategies.ForexTrader import ForexTrader


class DNN(ForexTrader):
    def __init__(
        self,
        conf_file: string,
        instrument: string,
        bar_length: string,
        units: int,
        duration: int,
        model: string = None,
        pkl: string = None,
        lags=5,
    ):
        self.model = None
        self.mean = None
        self.std = None
        self.lags = lags
        self.load_model(model, pkl)
        super().__init__(conf_file, instrument, bar_length, units, duration)

    def load_model(self, model_path: string, pkl_path: string):
        self.model = keras.models.load_model(model_path)
        params = pickle.load(open(pkl_path, "rb"))
        self.mean = params["mean"]
        self.std = params["std"]

    def define_strategy(self):
        df = self.raw_data.copy()

        df = df.append(self.tick_data)
        df["returns"] = calc.returns(df[self.instrument])
        df["dir"] = calc.dir(df.returns)
        df["sma"] = calc.sma_crossover(df[self.instrument])
        df["mean_reversion"] = calc.mean_reversion(df[self.instrument])
        df["min"] = calc.min(df[self.instrument])
        df["max"] = calc.min(df[self.instrument])
        df["mom"] = calc.mom(df.returns)
        df["vol"] = calc.volume(df.returns)
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
        ]

        for f in features:
            for lag in range(1, self.lags + 1):
                col = "{}_lag_{}".format(f, lag)
                df[col] = df[f].shift(lag)
                cols.append(col)
        df.dropna(inplace=True)

        df_s = (df - self.mean) / self.std

        df["prob"] = self.model.predict(df_s[cols])

        df = df.loc[self.start_time :].copy()
        df["position"] = np.where(df.prob < 0.47, -1, np.nan)
        df["position"] = np.where(df.prob > 0.53, 1, df.position)
        df["position"] = df.position.ffill().fillna(0)

        self.data = df.copy()
