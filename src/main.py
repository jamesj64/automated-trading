from strategies import ModdedMACD

trader = ModdedMACD.ModdedMACD(
    conf_file="oanda.cfg",
    instrument="EUR_USD",
    bar_length="1min",
    units=1000,
    duration=100,
)
