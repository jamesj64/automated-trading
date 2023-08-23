import time
import string
import numpy as np
import pandas as pd
from tpqoa import tpqoa
from datetime import datetime, timedelta


class ForexTrader(tpqoa):
    def __init__(
        self,
        conf_file: string,
        instrument: string,
        bar_length: string,
        units: int,
        duration: int,
        trading_hours: int * int = (0, 23),
    ):
        super().__init__(conf_file)
        self.instrument = instrument
        self.bar_length = pd.to_timedelta(bar_length)
        self.tick_data = pd.DataFrame()
        self.raw_data = None
        self.data = None
        self.last_bar = None
        self.units = units
        self.position = 0
        self.profits = []
        self.start_time = datetime.utcnow()
        self.end_time = self.start_time + timedelta(minutes=duration)

        self.trading_hours = trading_hours

        self.start_trading()

    def start_trading(
        self, days=5, max_attempts=None, wait=15, wait_increase=0
    ):  # Error Handling
        attempt = 0
        success = False
        while True:
            try:
                self.get_most_recent(days)
                self.stream_data(self.instrument)
            except Exception as e:
                print(e, end=" | ")
            else:
                success = True
                break
            finally:
                attempt += 1
                print("Attempt: {}".format(attempt), end="\n")
                if not success:
                    if max_attempts is not None and attempt >= max_attempts:
                        print("Max Attempts Reached!")
                        try:  # try to terminate session
                            time.sleep(wait)
                            self.terminate_session(
                                cause="Unexpected Session Stop (too many errors)."
                            )
                        except Exception as e:
                            print(e, end=" | ")
                            print("Could not terminate session properly!")
                        finally:
                            break
                    else:  # try again
                        time.sleep(wait)
                        wait += wait_increase
                        self.tick_data = pd.DataFrame()

    def set_trading_hours(self, trading_hours: int * int):
        self.trading_hours = trading_hours

    def terminate_session(self, cause: string):
        self.stop_stream = True
        if self.position != 0:
            close_order = self.create_order(
                self.instrument,
                units=-self.position * self.units,
                suppress=True,
                ret=True,
            )
            self.report_trade(close_order, "GOING NEUTRAL")
            self.position = 0
        print(cause, end=" | ")

    def close_open_position(self):
        if self.position == -1:
            order = self.create_order(
                self.instrument, self.units, suppress=True, ret=True
            )
            self.report_trade(order, "GOING NEUTRAL")
        elif self.position == 1:
            order = self.create_order(
                self.instrument, -self.units, suppress=True, ret=True
            )
            self.report_trade(order, "GOING NEUTRAL")
        print("\nSession Over.")
        self.position = 0

    def get_most_recent(self, days=5):
        time.sleep(1)
        print("~" * 50)
        print("Trying to merge...")
        print("Need below {} seconds".format(self.bar_length.seconds))
        now = datetime.utcnow()
        now = now - timedelta(microseconds=now.microsecond)
        past = now - timedelta(days=days)
        df = (
            self.get_history(
                instrument=self.instrument,
                start=past,
                end=now,
                granularity="S5",
                price="M",
                localize=True,
            )
            .c.dropna()
            .to_frame()
        )
        df.rename(columns={"c": self.instrument}, inplace=True)
        m_max = df.resample(self.bar_length, label="right").max().dropna()
        m_min = df.resample(self.bar_length, label="right").min().dropna()
        df = df.resample(self.bar_length, label="right").last().dropna().iloc[:-1]
        df["high"] = m_max
        df["low"] = m_min
        self.raw_data = df.copy()
        self.last_bar = self.raw_data.index[-1]
        print(
            "Seconds: {}".format(
                (pd.to_datetime(datetime.utcnow()) - self.last_bar).seconds
            )
        )
        if pd.to_datetime(datetime.utcnow()) - self.last_bar >= self.bar_length:
            print("Ensure that this is running during trading hours...")
            self.get_most_recent()
        else:
            print("Successfully Merged!")
            print("~" * 50)

    def on_success(self, t_time, bid, ask):
        recent_tick = pd.to_datetime(t_time).replace(tzinfo=None)

        print("Ticks: {}".format(self.ticks), end="\r", flush=True)

        if recent_tick >= self.end_time:
            self.terminate_session(cause="Scheduled Termination.")
            return

        df = pd.DataFrame({self.instrument: (ask + bid) / 2}, index=[recent_tick])
        self.tick_data = pd.concat([self.tick_data, df])

        (ts, te) = self.trading_hours

        curHour = (
            datetime.utcnow().tz_localize("UTC").tz_convert("America/New_York").hour
        )

        if (
            recent_tick - self.last_bar > self.bar_length
            and curHour >= ts
            and curHour <= te
        ):
            self.resample_and_join()
            self.define_strategy()
            self.execute_trades()

    def resample_and_join(self):
        new_row = (
            self.tick_data.resample(self.bar_length, label="right")
            .last()
            .ffill()
            .iloc[:-1]
        )
        new_row["high"] = (
            self.tick_data.resample(self.bar_length, label="right")
            .max()
            .ffill()
            .iloc[:-1]
        )
        new_row["low"] = (
            self.tick_data.resample(self.bar_length, label="right")
            .min()
            .ffill()
            .iloc[:-1]
        )
        self.raw_data = pd.concat([self.raw_data, new_row])
        self.tick_data = self.tick_data.iloc[-1:]
        self.last_bar = self.raw_data.index[-1]

    def define_strategy(self):  # "strategy-specific"
        # ONLY FOR BASE CLASS. DO NOT KEEP FOLLOWING LINES WHEN OVERRIDING
        df = self.raw_data.copy()
        df["position"] = 0
        self.data = df.copy()
        pass

    def execute_trades(self):
        if self.data["position"].iloc[-1] == 1:
            if self.position == 0:
                order = self.create_order(
                    self.instrument, self.units, suppress=True, ret=True
                )
                self.report_trade(order, "GOING LONG")
            elif self.position == -1:
                order = self.create_order(
                    self.instrument, self.units * 2, suppress=True, ret=True
                )
                self.report_trade(order, "GOING LONG")
            else:
                print("\nStaying long...")
            self.position = 1
        elif self.data["position"].iloc[-1] == -1:
            if self.position == 0:
                order = self.create_order(
                    self.instrument, -self.units, suppress=True, ret=True
                )
                self.report_trade(order, "GOING SHORT")
            elif self.position == 1:
                order = self.create_order(
                    self.instrument, -self.units * 2, suppress=True, ret=True
                )
                self.report_trade(order, "GOING SHORT")
            else:
                print("\nStaying short...")
            self.position = -1
        elif self.data["position"].iloc[-1] == 0:
            if self.position == -1:
                order = self.create_order(
                    self.instrument, self.units, suppress=True, ret=True
                )
                self.report_trade(order, "GOING NEUTRAL")
            elif self.position == 1:
                order = self.create_order(
                    self.instrument, -self.units, suppress=True, ret=True
                )
                self.report_trade(order, "GOING NEUTRAL")
            else:
                print("\nStaying neutral...")
            self.position = 0

    def report_trade(self, order, going):
        time = order["time"]
        units = order["units"]
        price = order["price"]
        pl = float(order["pl"])
        self.profits.append(pl)
        cumpl = sum(self.profits)
        print("\n" + 50 * "~")
        print("{} | {}".format(time, going))
        print(
            "{} | units = {} | price = {} | P&L = {} | Cum P&L = {}".format(
                time, units, price, pl, cumpl
            )
        )
        print(50 * "~")
