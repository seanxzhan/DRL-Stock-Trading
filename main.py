import numpy as np
import tensorflow as tf
from agent import PolicyGradientAgent
from stock_env import StockEnv, discount
from preprocess import get_data
from random import randint
from visual_helpers import visualize_linegraph, visualize_portfolio



def train(train_data, model, num_rand_stocks, all_tickers):
    """
    the train function, train the model for an entire epoch

    :param train_data: the preprocessed training data, of shape [num_stocks, num_days, datum_size]
    :param model: the RL agent
    :param num_rand_stocks: number of random stocks to feed to the network in each episode
    :param all_tickers: all the preprocessed tickers

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
        episode_input = train_data[:, start:end, :]
        
        # pick num_stocks random stocks
        # exclude cash when we randomly pick stocks (cash is the last element of all_tickers)
        rand_stock_indices = np.random.choice(len(all_tickers) - 1, num_rand_stocks, replace=False)
        rand_stock_indices = tf.reshape(rand_stock_indices, (len(rand_stock_indices), 1))
        episode_input = tf.gather_nd(episode_input, rand_stock_indices)

        # randomize starting date, keep at least one chunk of num_days left
        # past_num = model.past_num
        # offset = 2 * past_num
        # rand_start = randint(0, max_episode_days - past_num - offset)

        # randomize starting date in each episode
        rand_start = randint(0, int(max_episode_days / 5))
        episode_input = episode_input[:, rand_start:max_episode_days,:]

        env = StockEnv(episode_input, train_tickers)

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
    
    # toggle this
    randomize_input_stocks = False
    num_rand_stocks = 5 if randomize_input_stocks else num_stocks

    # creating model
    model = PolicyGradientAgent(datum_size, num_rand_stocks, past_num)


    # training
    for i in range(NUM_EPOCH):
        print(f'EPOCH: --------------------------------{i}')
        start_day = randint(0, num_days - model.past_num - model.batch_size)  # TODO: inefficient usage of data?
        sample = train_data[:, start_day:, :]
        epoch_loss = train(train_data, model, num_rand_stocks, x_tickers)  # change to sample for random initial time step
        print(f"loss list for epoch {i} is {epoch_loss}")


    # testing
    test(test_data, model, y_tickers)
    print("END")


if __name__ == '__main__':
    main()
