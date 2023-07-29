from strategies import SMACrossover

trader = SMACrossover.SMACrossover(
    conf_file="oanda.cfg",
    instrument="EUR_USD",
    bar_length="1min",
    units=1000,
    duration=15,
)
