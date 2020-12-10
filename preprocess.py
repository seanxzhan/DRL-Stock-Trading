import yfinance as yf
import tensorflow as tf


def get_data(all_tickers, training_ratio=0.8, validating_ratio=0.1):
    """
    Import data from yfinance and split data into training set and testing set

    :param all_tickers: a list of all the Stock names whose data needs to be retrived
    :param training_ratio: the ratio of the training set in the whole dataset

    :returns: (train_data, test_data, all_tickers)
              Each data tensor has dimension (num_stocks, num_days, datum_size)
              all_tickers is a list with "CASH" appended to the end of input tickers
    """
    str_tickers = ' '.join(all_tickers)
    num_stocks = len(all_tickers)
    datum_size = 5      # open, high, low, adjusted close, volume

    # start, end, interval can be changed
    daily_data = yf.download(
        tickers = str_tickers,
        start = "2010-01-01",
        end = "2020-01-01",
        interval = "1d", 
        group_by = "ticker"
    )

    labels = list(filter(lambda x: x[1] in ['Close'], daily_data.columns.values))
    daily_data = daily_data.drop(columns=labels)
    daily_data = tf.convert_to_tensor(daily_data, dtype=tf.float32)
    num_days = daily_data.shape[0]
    daily_data = tf.reshape(daily_data, [num_days, num_stocks, datum_size])
    daily_data = tf.transpose(daily_data, perm=[1, 0, 2]) 
    # now daily_data has dimension [num_stocks, num_days, datum_size]

    train_cutoff = int(training_ratio * num_days)
    valid_cutoff = int(validating_ratio * num_days) + train_cutoff
    train_data = daily_data[:, 0:train_cutoff, :]
    valid_data = daily_data[:, train_cutoff:valid_cutoff, :]
    test_data = daily_data[:, valid_cutoff:num_days, :]

    all_tickers.append("CASH")
    return train_data, valid_data, test_data, all_tickers
