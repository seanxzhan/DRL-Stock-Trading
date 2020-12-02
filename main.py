import numpy as np
import tensorflow as tf
from agent import PolicyGradientAgent
from stock_env import StockEnv, discount
from preprocess import get_data
from random import randint
from visual_helpers import visualize_linegraph, visualize_portfolio


def train(train_data, model, train_tickers):
    """
    the train function, train the model for an entire epoch

    :param train_data: the preprocessed training data, of shape [num_stocks, num_days, datum_size]
    :param model: the RL agent

    :return to be decided
    """
    episode_max_days = randint(190, 210)  # the maximum number of days of an episode
    episode_max_days = 70
    num_days = train_data.shape[1]
    loss_list = []
    for episode in range(0, int(num_days / episode_max_days)):
        if episode % 10 == 0: print(f"Training batch {episode}")
        start = episode * episode_max_days
        end = start + episode_max_days
        batch_input = train_data[:, start:end, :]
        env = StockEnv(batch_input, train_tickers)
        with tf.GradientTape() as tape:
            states, actions, rewards = env.generate_episode(model)
            discounted_rewards = discount(rewards)
            model.remember(states, actions, discounted_rewards)
            repl_states, repl_actions, repl_discounted_rewards = model.experience_replay()
            model_loss = model.loss(repl_states, repl_actions, repl_discounted_rewards)
        gradients = tape.gradient(model_loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        loss_list.append(model_loss)  # reward at end of batch
    return list(loss_list)


def test(test_data, model, tickers):
    """
    the test function: DOCSTRING TO BE WRITTEN
    """
    print("testing")
    env = StockEnv(test_data, tickers, is_testing=True)
    states, actions, rewards = env.generate_episode(model)

    print(f'final cash: {rewards[-1]}')


def main():
    """
    probabilities = NaN error occurs occasionally.
    """
    NUM_EPOCH = 3

    # pre-process data
    train_tickers = ["AAPL", "AMZN", "MSFT", "INTC", "REGN"]
    test_tickers = ["ADBE", "DIS", "JNJ", "HON", "PFE"]
    train_data, _, x_tickers = get_data(train_tickers)  # data: (num_stock, num_days, datum_size)
    test_data, _, y_tickers = get_data(test_tickers)
    test_data = test_data[:, 0:200, :]
    num_stocks, num_days, datum_size = train_data.shape
    past_num = 50

    # creating model
    model = PolicyGradientAgent(datum_size, num_stocks, past_num)

    # training
    for i in range(NUM_EPOCH):
        print(f'EPOCH: --------------------------------{i}')
        start_day = randint(0, num_days - model.past_num - model.batch_size)  # TODO: inefficient usage of data?
        sample = train_data[:, start_day:, :]
        epoch_loss = train(train_data, model, x_tickers)  # change to sample for random initial time step
        print(f"loss list for epoch {i} is {epoch_loss}")

    # testing
    test(test_data, model, y_tickers)
    print("END")


if __name__ == '__main__':
    main()
