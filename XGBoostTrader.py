import backtrader as bt


"""
This is a skeleton for the XGBoost strategy.
This will hopefully be built to purely execute trades based on XGBoost predictions.
e.g. price predicted to go up 20% in 6 months -> BUY
if the accuracy of XGBoost is not too high (<80%) then it will be used as an extra indicator
in combination with bollinger bands, moving averages and rsi.
"""

class XGTrader(bt.strategy):


    def __init__(self):
        # ToDo: Load XGBoost model from file
        # Set up required variables and data structs (still not ironed these out yet)
        pass

    # this is the backtraders main function, it is called when new data is available
    # e.g. the next row in a dataset of trade data
    def next(self):
        # ToDo: use XGBoost to predict 6 month price
        # ToDo: buy based on indicator
        # ToDo: store metrics e.g. portfolio value, gain, loss to eval strategy
        pass

    