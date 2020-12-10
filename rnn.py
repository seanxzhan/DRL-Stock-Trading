import tensorflow as tf
import numpy as np
from agent import save_model, load_model
from preprocess import get_data


def rnn_create():
    """
    building a RNN model.
    This model takes in a sequence of historical data, and learns to estimate the price changes the stock will have
    in the current time step

    return: model, an initialized tf GRU model.
    """
    H1 = 16
    H2 = 32
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.01, decay_rate=0.97,
                                                                   decay_steps=300)
    # learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[10, 30, 60, 100],
    #                                                                         values=[0.01, 0.005, 0.003, 0.002, 0.001])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.GRU(H1))  # out shape: (batch_sz, H1)
    model.add(tf.keras.layers.Dense(H2, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.1))
    model.add(tf.keras.layers.Dense(5))
    model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanAbsolutePercentageError(),
                  metrics=[tf.keras.metrics.MeanAbsolutePercentageError()])
    return model


def rnn_train(model, train_x, train_y, num_epoch):
    """
    a function to train the RNN model for
    :param model: the RNN model to be trained
    :param train_x: the train inputs, of shape (num_examples, past_num, datum_size)
    :param train_y: the train labels, of shape (num_examples, datum_size)
    :param num_epoch: number of epochs to train for

    :return history: the training history of training the model
    """
    history = model.fit(train_x, train_y, batch_size=32, epochs=num_epoch, validation_split=0.1)
    return history


def rnn_test(model, test_x, test_y):
    """
    evaluate the performance of the RNN model
    :param model: the RNN model to be trained
    :param test_x: the test inputs, of shape (num_examples, past_num, datum_size)
    :param test_y: the test labels, of shape (num_examples, datum_size)

    return: result of training
    """
    result = model.evaluate(test_x, test_y, batch_size=32)
    return result


def process_data(data, past_num, x_file_path, y_file_path):
    """
    process the data into (x,y) pairs needed for training and testing
    :param data: either train_data or test_data, of shape (num_stocks, num_days, datum_size)
    :param past_num: the past number of days used to predict the current stock price changes
    :param x_file_path: save the result x to this file path
    :param y_file_path: save the result y to this file path

    :return (x, y): x of shape (num_examples, past_num, datum_size)
                    y of shape (num_examples, datum_size)
    """
    num_stocks, num_days, datum_size = data.shape
    num_examples = num_days - past_num
    x = []
    y = []
    for i_stock in range(num_stocks):
        stock = data[i_stock]  # (num_days, datum_size)
        for i_example in range(num_examples):
            x.append(stock[i_example: i_example+past_num, :])
            y.append(stock[i_example + past_num, :] - stock[i_example + past_num - 1, :])
    x = np.array(x)
    y = np.array(y)
    assert x.shape == (num_examples * num_stocks, past_num, datum_size), "x not the right size"
    assert y.shape == (num_examples * num_stocks, datum_size), "y not the right size"
    np.save(x_file_path, x)
    np.save(y_file_path, y)
    return x, y


def main():
    """
    the main function for creating, training, saving, and testing the RNN model

    return: nothing
    """
    SAVE = True  # whether to save the transformer model parameters
    RESUME = False  # resume from a saved model
    SAVED_DATA = True  # whether the preprocessed data is saved
    PATH_TO_MODEL = "rnn_model"
    # tickers = ["AAPL", "AMZN", "GOOGL", "MSFT", "CVS", "DIS", "FDX", "JPM", "REGN", "WMT", "JNJ", "HON"]  # all stocks
    tickers = ["AAPL", "AMZN"]
    train_data, test_data, _ = get_data(tickers)  # data of shape (num_stocks, num_days, datum_size)
    past_num = 50  # the number of days the model looks at to predict the next stock price

    # create model
    if RESUME:
        model = load_model(PATH_TO_MODEL)
    else:
        model = rnn_create()

    # pre-process data
    print("========== beginning to pre-process data ==========")
    train_x_path = "data/new_rnn_train_x.npy"
    train_y_path = "data/new_rnn_train_y.npy"
    test_x_path = "data/new_rnn_test_x.npy"
    test_y_path = "data/new_rnn_test_y.npy"
    if SAVED_DATA:
        print("loading saved data")
        train_x, train_y = np.load(train_x_path), np.load(train_y_path)
        test_x, test_y = np.load(test_x_path), np.load(test_y_path)
    else:
        train_x, train_y = process_data(train_data, past_num, train_x_path, train_y_path)
        test_x, test_y = process_data(test_data, past_num, test_x_path, test_y_path)
    print("========== finished pre-process data ==========")

    # train the model
    NUM_EPOCH = 300
    print("========== begin training ==========")
    train_history = rnn_train(model, train_x, train_y, NUM_EPOCH)
    print("========== finish training ==========")
    if SAVE:
        save_model(model, PATH_TO_MODEL)
        print("====== saved model =======")

    # evaluate the model
    print("========== begin testing ==========")
    result = rnn_test(model, test_x, test_y)
    print("printing results")
    print(model.metrics_names)
    print(result)
    print("========== finished testing ==========")


if __name__ == "__main__":
    main()
