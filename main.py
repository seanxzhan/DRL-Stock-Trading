import numpy as np
import tensorflow as tf
from agent import PolicyGradientAgent
from stock_env import *


def train(train_data, model):
    """
    the train function, train the model for an entire epoch

    :param train_data: the preprocessed training data, of shape [num_data_points, state_size]
    :param model: the RL agent

    :return to be decided
    """
    batch_size = model.batch_size
    # TODO: 0) batch the train_data, and for each batch:
    # TODO: 1) generate an episode, put it in the memory buffer (use generate_episode() from stock_env.py)
    # TODO: 2) sample a batch of (state, action, reward) from the memory buffer
    # TODO: 3) compute the discounted rewards
    # TODO: 4) compute the loss, run back prop on the model
    pass


def test(test_data, model):
    """
    the test function: DOCSTRING TO BE WRITTEN
    """
    # TODO: use some kind of evaluation metric to determine how good our model is
    pass


def main():
    """
    where everything comes together
    """
    NUM_EPOCH = None
    # TODO: parse cmd line arguments if needed
    # TODO: import preprocessed data from file in the current directory
    # TODO: decide if train from beginning, or load a previously trained model
    # TODO: create an instance of the agent
    # TODO: train
    # TODO: test


if __name__ == '__main__':
    main()
