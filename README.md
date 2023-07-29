# Automated Trader

## Overview
An automated trading bot framework with multiple strategies for forex with Oanda built with [tpqoa](https://github.com/yhilpisch/tpqoa). I will update this repository shortly to include a general framework for backtesting both vectorized and iterative/event-driven implemented strategies. This is in no way an endorsement to engage in live trading and this repository should not be considered as financial advice. You should not run this code with a live Oanda account.

Some of the default strategies include simple momentum/contrarian, simple moving average crossover, exponential moving average crossover, mean reversion (bollinger bands), and MACD. I plan on adding more strategies in the near future.

Note that the Jupyter files are not required and are actually excluded from the Docker image.

Some of the technologies used in making this project include: Python, Pandas, Docker, Jupyter, NumPy, MatPlotLib, SciKit.

## Usage
Ensure that in your working directory you have the Oanda configuration file `oanda.cfg` filled out with your account/api information. The config file should be filled out as follows:
```
[oanda]
account_id = XXX-XXX-XXXXXXXX-XXX
access_token = XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
account_type = practice (default) or live
```
After creating and configuring `oanda.cfg`, you can run

```
pip install -r requirements.txt
python ./src/main.py
```

or build and run the Docker image with

```
docker build -t automated-trader .
docker run --rm -it --name automated-trader automated-trader
```