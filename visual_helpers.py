import numpy as np
import pandas as pd
from stock_env import *
from preprocess import get_data
import matplotlib.pyplot as plt
import datetime

def bad_visualization():
    train_data, test_data, daily_data = get_data()
    x = daily_data[0, 1, :]
    print(x.shape)
    for i in range(4):
        plt.plot(daily_data[i, 1, :])
    plt.show()

def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys())


def visualize_linegraph(rewards):
    """
    param: rewards (episode_length, )

    create a line graph of a 1D vector
    can be used to visualize stock prices or rewards
    """
    fig = plt.figure()
    plt.title('Rewards')
    plt.plot(rewards)
    plt.show()


def visualize_portfolio(portfolio, tickers):
    """
    param: portfolio [num_stocks + 1, episode_length]
            portfolio[i][d] gives the price of stock i on day d
            episode_length can be any number if you don't want to visualize the entire episode
    """

    # portfolio = {
    #     "a": [1, 2, 3, 2, 1],
    #     "b": [2, 3, 4, 3, 1],
    #     "c": [3, 2, 1, 4, 2],
    #     "d": [5, 9, 2, 1, 8],
    #     "e": [1, 3, 2, 2, 3],
    #     "f": [4, 3, 1, 1, 4],
    # }
    dict_portfolio = {}
    num_stocks = len(tickers) - 1 #excluding cash

    for i in range(num_stocks + 1):
        str = tickers[i]
        dict_portfolio[str] = portfolio[i]

    print("portfolio dict in vis:", dict_portfolio)

    fig, ax = plt.subplots()
    bar_plot(ax, dict_portfolio, total_width=.8, single_width=.9)
    plt.show()


def unit_test_visualization():
    train, test, _ = get_data()
    print(f'shape: {train.shape}')
    print(train[:, :10, :])
    visualize_linegraph(train[0, :, 0])
    visualize_linegraph(train[1, :, 0])

    tickers = ['AAPL', 'AMZN', 'MSFT', 'INTC', 'REGN', 'CASH']
    portfolio = [[1, 2, 3, 2, 1], [2, 3, 4, 3, 1], [3, 2, 1, 4, 2], [5, 9, 2, 1, 8], [1, 3, 2, 2, 3], [4, 3, 1, 1, 4]]
    visualize_portfolio(portfolio, tickers)

    # test, train = get_data()
    # num_stocks, num_days, datum_size = test.shape
    # print(num_stocks)
    # print(num_days)
    # print(datum_size)
