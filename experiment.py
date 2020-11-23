import numpy as np
import pandas as pd
from stock_env import *
from preprocess import get_data
import matplotlib.pyplot as plt
from stock_env import numberToBase

def bad_visualization():
    train_data, test_data, daily_data = get_data()
    x = daily_data[0, 1, :]
    print(x.shape)
    for i in range(4):
        plt.plot(daily_data[i, 1, :])
    plt.show()


if __name__ == '__main__':
    x = get_data()
    train, test, daily = get_data()
    print(train.shape)
    print(test.shape)



