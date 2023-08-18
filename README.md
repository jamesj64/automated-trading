# Automated Trader

## Overview
An automated trading bot framework with multiple strategies for forex with Oanda built with [tpqoa](https://github.com/yhilpisch/tpqoa). This repository also includes a general framework for backtesting both vectorized and iterative/event-driven implemented strategies. This is in no way an endorsement to engage in live trading and this repository should not be considered financial advice. You should not run this code with a live Oanda account.

Some of the default strategies include simple momentum/contrarian, simple moving average crossover, exponential moving average crossover, mean reversion (bollinger bands), and MACD. I have also trained a DNN with Tensorflow/Keras to predict the direction of the market. I plan on adding more strategies in the near future.

Some of the technologies used in making this project include: Python, Pandas, Tensorflow/Keras, Docker, Jupyter, NumPy, MatPlotLib, and SciKit.

## Usage
Ensure that in your root directory you create an Oanda configuration file `oanda.cfg` filled out with your account/api information. The config file should be filled out as follows:
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

To see a demo, check [this](src/main.ipynb) out.