# Project Overview
This project focuses on forecasting stock price movements using 3 distinct machine learning models:

- XGBoost
- Linear Regression
- LSTM (Long Short-Term Memory networks)

My aim is to assess the accuracy of these models both for accuracy in predicting prices and later by using them as trading strategies, executing trades based off of their price predictions.

# Models
### XGBoost
**Usage:**
XGBoost is applied to longer-term price predictions. This model will use daily stock data and will forecast stock prices months into the future (testing 1,3,6 month intervals).

### Linear Regression
**Usage:**
Linear Regression will be used for medium-term price prediction. Although Linear Regression is simpler than the other models, it is still good to test such a model for this task. If the relationships between the data and target variable are linear then this model can outperform the rest, it is also less resource intensive and easier to train.

### LSTM
**Usage:**
LSTMs are ideal for short-term, intra-day price prediction. Their ability to capture sequential data patterns and temporal dependencies makes them well-suited for forecasting price fluctuations over hours or single trading sessions.


## Conclusion
By applying these models across different timescales, this project aims to determine the most effective model for stock price prediction.
**Note:** This project uses the Kaggle stock dataset