import numpy as np
import tensorflow as tf
from agent import PolicyGradientAgent
from stock_env import *
from visual_helpers import visualize_linegraph, visualize_portfolio


def train(train_data, model, past_num, num_rand_stocks, all_tickers):
    """
    the train function, train the model for an entire epoch

    :param train_data: the preprocessed training data, of shape [num_stocks, num_days, datum_size]
    :param model: the RL agent
    :param past_num: past number of days to consider when making a trading decision
    :param num_rand_stocks: number of random stocks to feed to the network in each episode
    :param all_tickers: all the preprocessed tickers

    :return to be decided
    """
    max_episode_days = 120
    total_num_days = train_data.shape[1]
    # TODO: 0) batch the train_data, and for each batch:
    # TODO: 1) generate an episode, put it in the memory buffer (use generate_episode() from stock_env.py)
    # TODO: 2) sample a batch of (state, action, reward) from the memory buffer
    # TODO: 3) compute the discounted rewards
    # TODO: 4) compute the loss, run back prop on the model
    for episode in range(0, int(total_num_days / max_episode_days)):
        if episode % 10 == 0: print("Training episode #{}".format(episode))
        start = episode * max_episode_days
        end = start + max_episode_days
        episode_input = train_data[:, start:end, :]

        # pick num_stocks random stocks
        # exclude cash when we randomly pick stocks (cash is the last element of all_tickers)
        rand_stock_indices = np.random.choice(len(all_tickers) - 1, num_rand_stocks, replace=False)
        rand_stock_indices = tf.reshape(rand_stock_indices, (len(rand_stock_indices), 1))
        episode_input = tf.gather_nd(episode_input, rand_stock_indices)

        # randomize starting date, keep at least one chunk of num_days left
        # offset = 2 * past_num
        # rand_start = randint(0, max_episode_days - past_num - offset)

        # randomize starting date in each episode
        rand_start = randint(0, int(max_episode_days / 5))
        episode_input = episode_input[:, rand_start:max_episode_days,:]

        env = StockEnv(episode_input, all_tickers)
        with tf.GradientTape() as tape:
            states, actions, rewards = env.generate_episode(model)
            discounted_rewards = discount(rewards)
            model.remember(states, actions, discounted_rewards)
            repl_states, repl_actions, repl_discounted_rewards = model.experience_replay()
            
            model_loss = model.loss(repl_states, repl_actions, repl_discounted_rewards)
            print("Loss: {}".format(model_loss))

        gradients = tape.gradient(model_loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    pass


def test(test_data, model, tickers):
    """
    the test function: DOCSTRING TO BE WRITTEN
    """
    # TODO: use some kind of evaluation metric to determine how good our model is
    # TODO: 0) batch the train_data, and for each batch:
    # TODO: 1) generate an episode, put it in the memory buffer (use generate_episode() from stock_env.py)
    # TODO: 2) sample a batch of (state, action, reward) from the memory buffer
    # TODO: 3) compute the discounted rewards
    # TODO: 4) compute the loss, run back prop on the model

    print("testing")
    env = StockEnv(test_data, all_tickers=tickers, is_testing = True)
    states, actions, rewards = env.generate_episode(model)

    print(f'final cash: {rewards[-1]}')


def main():
    """
    probabilities = NaN error occurs occasionally.
    """
    NUM_EPOCH = 1
    # TODO: parse cmd line arguments if needed
    # TODO: import preprocessed data from file in the current directory
    # TODO: decide if train from beginning, or load a previously trained model
    # TODO: create an instance of the agent
    # TODO: train
    # TODO: test

    train_data, test_data, tickers = get_data()  # data: (num_stock, num_days, datum_size)
    num_stocks, num_days, datum_size = test_data.shape
    past_num = 30

    # toggle this
    randomize_input_stocks = False
    if randomize_input_stocks:
        num_rand_stocks = 5
    else:
        num_rand_stocks = num_stocks

    model = PolicyGradientAgent(datum_size, num_rand_stocks, past_num)

    for i in range(NUM_EPOCH):
        print(f'EPOCH: --------------------------------{i}')
        train(train_data, model, past_num, num_rand_stocks, tickers)

    test(test_data, model, tickers)
    print("END")


if __name__ == '__main__':
    main()
