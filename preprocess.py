# TODO: import the yahoo finance data
# TODO: process the data
# TODO: save the data in this directory (so that we don't need to run preprocess every time we try to use the model)
# can use np's save function, pickle's save function, ....

# for example, save a dictionary {'train': ...,  'test': ...}
# and value of 'train' could be a [num_data_points, state_size] matrix, where each row represents one data point,
# containing info like opening price, closing price, technical indicators, ...
# similar for 'test'

# or to give us more flexibility later, 'train' could also be a nested dict, 'train' --> year --> month --> day --> data

import yfinance as yf
import tensorflow as tf

def get_data():
    """
    Import data from yfinance and split data into training set and testing set

    :returns: A tuple of tensors (training set, testing set). 
              Each tensor has dimension [num_stocks, state_size, num_days]
    """
    num_stocks = 5      # AAPL AMZN MSFT INTC REGN
    state_size = 6      # open, high, low, close, adjusted close, volume
    train_to_test_ratio = 0.9

    daily_data = yf.download(
        tickers = "AAPL AMZN MSFT INTC REGN",
        start = "2000-01-01",
        end = "2020-01-01",
        interval = "1d",
        group_by = "ticker"
    )

    daily_data = tf.convert_to_tensor(daily_data, dtype=tf.float32)
    num_days = daily_data.shape[0]
    daily_data = tf.reshape(tf.transpose(daily_data), [num_stocks, state_size, num_days])

    cutoff = int(train_to_test_ratio * num_days)
    print(cutoff)
    train_data = daily_data[:, :, 0:cutoff]
    test_data = daily_data[:, :, cutoff:num_days]
    return train_data, test_data

# data_hourly = yf.download(
#     tickers = "AAPL", # it seems like yfinance doesn't support hourly data for multiple tickers
#     start = "2018-11-21", # must be within the last 730 days
#     end = "2020-11-19",
#     interval = "1d",
#     group_by = "ticker"
# )
# print(data_hourly)
# data_hourly = tf.convert_to_tensor(data_hourly)
# print(data_hourly)