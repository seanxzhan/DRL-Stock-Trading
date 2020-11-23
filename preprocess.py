import yfinance as yf
import tensorflow as tf

def get_data():
    """
    Import data from yfinance and split data into training set and testing set

    :returns: A tuple of tensors (training set, testing set). 
              Each tensor has dimension [num_stocks, num_days, state_size]
    """
    all_tickers = ["AAPL", "AMZN", "MSFT", "INTC", "REGN"]
    str_tickers = ' '.join(all_tickers)
    num_stocks = len(all_tickers)    
    state_size = 5      # open, high, low, adjusted close, volume
    train_to_test_ratio = 0.9

    # start, end, interval can be changed
    daily_data = yf.download(
        tickers = str_tickers,
        start = "2000-01-01",
        end = "2020-01-01",
        interval = "1d", 
        group_by = "ticker"
    )

    labels = list(filter(lambda x: x[1] == 'Close', daily_data.columns.values))
    daily_data = daily_data.drop(columns=labels)
    daily_data = tf.convert_to_tensor(daily_data, dtype=tf.float32)
    num_days = daily_data.shape[0]
    daily_data = tf.reshape(daily_data, [num_days, num_stocks, state_size])
    daily_data = tf.transpose(daily_data, perm=[1, 0, 2]) 
    # now daily_data has dimension [num_stocks, num_days, state_size]

    cutoff = int(train_to_test_ratio * num_days)
    train_data = daily_data[:, 0:cutoff, :]
    test_data = daily_data[:, cutoff:num_days, :]
    #TODO: necessary to save this data?
    return train_data, test_data

get_data()

# IF WE DECIDE TO DO HOURLY DATA:
# data_hourly = yf.download(
#     tickers = "AAPL", # it seems like yfinance doesn't support hourly data for multiple tickers
#     start = "2018-11-21", # must be within the last 730 days
#     end = "2020-11-19",
#     interval = "1d",
#     group_by = "ticker"
# )
# data_hourly = tf.convert_to_tensor(data_hourly)