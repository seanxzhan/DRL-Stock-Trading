import yfinance as yf
import tensorflow as tf


def get_data():
    """
    Import data from yfinance and split data into training set and testing set

    :returns: A tuple of tensors (training set, testing set). 
              Each tensor has dimension [num_stocks, num_days, datum_size]
    """
    num_stocks = 5      # AAPL AMZN MSFT INTC REGN
    state_size = 6      # open, high, low, close, adjusted close, volume
    train_to_test_ratio = 0.9

    # start, end, interval can be changed
    daily_data = yf.download(
        tickers="AAPL AMZN MSFT INTC REGN",
        start="2000-01-01",
        end="2020-01-01",
        interval="1d",
        group_by="ticker"
    )

    daily_data = tf.convert_to_tensor(daily_data, dtype=tf.float32)
    num_days = daily_data.shape[0]
    daily_data = tf.reshape(tf.transpose(daily_data), [
                            num_stocks, state_size, num_days])

    cutoff = int(train_to_test_ratio * num_days)
    train_data = daily_data[:, :, 0:cutoff]
    test_data = daily_data[:, :, cutoff:num_days]
    # TODO: necessary to save this data?
    return train_data, test_data, daily_data
    print(test_data.shape)
# IF WE DECIDE TO DO HOURLY DATA:
# data_hourly = yf.download(
#     tickers = "AAPL", # it seems like yfinance doesn't support hourly data for multiple tickers
#     start = "2018-11-21", # must be within the last 730 days
#     end = "2020-11-19",
#     interval = "1d",
#     group_by = "ticker"
# )
# data_hourly = tf.convert_to_tensor(data_hourly)
