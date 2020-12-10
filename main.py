import numpy as np
import tensorflow as tf
from agent import PolicyGradientAgent, save_model, load_model
from stock_env import StockEnv, discount
from preprocess import get_data
from random import randint
from visual_helpers import visualize_linegraph, visualize_portfolio


def train(train_data, model, tickers, randomize, num_rand_stocks=0, episode_max_days=200):
    """
    the train function, train the model for an entire epoch

    :param train_data: the preprocessed training data, of shape [num_stocks, num_days, datum_size]
    :param model: the model to be trained
    :param tickers: stock tickers corresponding to train_data, including "CASH"
    :param randomize: boolean indicating whether we have randomized stocks
    :param num_rand_stocks: number of stocks randomized in training
    :param episode_max_days: the maximum number of days of trading actions in an episode

    :return losses and rewards
    """
    num_days = train_data.shape[1]
    loss_list = []

    offset = model.past_num - 1 # extra days of price history needed at beginning
    start = 0  # start of price history slice (inclusive)
    end = start + episode_max_days + offset  # end of price history slice (exclusive)
    num_episodes = (num_days - offset) // episode_max_days

    # a list of total cash value
    rewards_list = []

    for episode in range(num_episodes):
        print(f"Training episode {episode+1} of {num_episodes}")

        if randomize:
            # the last element of tickers is "CASH", we don't include "CASH" in randomization
            rand_stock_indices = np.random.choice(len(tickers) - 1, num_rand_stocks, replace=False)
            # get randomly selected stock names
            episode_tickers = [tickers[index] for index in rand_stock_indices]
            episode_tickers.append("CASH")
            rand_stock_indices = tf.reshape(rand_stock_indices, (len(rand_stock_indices), 1))
            episode_input = tf.gather_nd(train_data, rand_stock_indices) 
        else:
            episode_input = train_data
            episode_tickers = tickers

        # Slice of pricing history to generate this episode on
        episode_input = episode_input[:, start:end, :]
        start += episode_max_days
        end += episode_max_days
        
        # (REMOVED) randomize starting date in each episode
        # rand_start = randint(0, int(episode_max_days / 5))
        # episode_input = episode_input[:, rand_start:episode_max_days,:]

        # Hyperparameters to be adjusted below:
        env = StockEnv(episode_input, episode_tickers, interest_annual=0.1, borrow_interest_annual=0.2, transaction_penalty=0.0001)

        with tf.GradientTape() as tape:
            states, actions, rewards = env.generate_episode(model)
            rewards_list.extend(rewards)
            discounted_rewards = discount(rewards)
            model.remember(states, actions, discounted_rewards)
            repl_states, repl_actions, repl_discounted_rewards = model.experience_replay()
            model_loss = model.loss(repl_states, repl_actions, repl_discounted_rewards)
        gradients = tape.gradient(model_loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        loss_list.append(model_loss.numpy())  # reward at end of batch
    return list(loss_list), rewards_list


def test(test_data, model, tickers, randomize, num_rand_stocks=0):
    """
    the test function: test agent on test_data
    if randomize == True, we know we have randomized stocks in training, 
    so we test on num_rand_stocks randomly selected stocks from test_data
    if otherwise, we used the entire test_data set

    :param test_data: the testing set
    :param model: the trained model
    :param tickers: stocks tickers corresponding to test_data, including "CASH"
    :param randomize: boolean indicating whether we have randomized stocks
    :param num_rand_stocks: number of stocks randomized in training
    """
    if randomize:
        # the last element of tickers is "CASH", we don't include "CASH" in randomization
        rand_stock_indices = np.random.choice(len(tickers) - 1, num_rand_stocks, replace=False)
        # get randomly selected stock names
        episode_tickers = [tickers[index] for index in rand_stock_indices]
        episode_tickers.append("CASH")
        rand_stock_indices = tf.reshape(rand_stock_indices, (len(rand_stock_indices), 1))
        # saving the randomization to a new variable so we don't mess w/ test_data
        episode_input = tf.gather_nd(test_data, rand_stock_indices) 
    else:
        episode_input = test_data
        episode_tickers = tickers

    env = StockEnv(episode_input, episode_tickers, is_testing=True)
    states, actions, rewards = env.generate_episode(model)
    min_testing_episode_len = 20
    while len(rewards) < min_testing_episode_len:
        print("test episode not long enough")
        states, actions, rewards = env.generate_episode(model)
    print(f'final portfolio total value: {rewards[-1]}')


def main():
    """
    Main
    """
    NUM_EPOCHS = 10
    RESUME = False
    SAVE = False
    RANDOMIZE = False
    VALIDATING = True

    # TODO
    # run stocks separately in each episode
    # individualy pass in stocks w/ evenly distributed initial cash

    train_tickers = ["AAPL", "AMZN", "GOOGL", "MSFT"]

    if RANDOMIZE:
        # add more stocks to train_tickers if we want
        train_tickers.extend(["CVS", "DIS", "FDX", "JPM"])
    test_tickers = ["REGN", "WMT", "JNJ", "HON"]
    train_data, valid_data_same_stocks, test_data_same_ticks, x_tickers = get_data(train_tickers) # train_data should be the same for both testing methods
    _, _, test_data_diff_ticks, y_tickers = get_data(test_tickers)

    # The following block is removed in order to increase the testing period.
    # testing_days = 200
    # # we select random testing days for test_data_diff_ticks and test_data_same_ticks
    # assert test_data_same_ticks.shape[1] == test_data_diff_ticks.shape[1]
    # assert test_data_diff_ticks.shape[1] > testing_days
    # rand_start = randint(0, int(( test_data_diff_ticks.shape[1] - testing_days ) / 5 ))
    # rand_end = rand_start + testing_days
    # # the testing days for both methods will be the same
    # test_data_diff_ticks = test_data_diff_ticks[:, rand_start:rand_end, :]
    # test_data_same_ticks = test_data_same_ticks[:, rand_start:rand_end, :]

    num_stocks, num_days, datum_size = train_data.shape
    past_num = 50
    
    # we can also toggle the number below
    num_rand_stocks = 4 if RANDOMIZE else num_stocks

    # creating model
    if RESUME:
        model = load_model('saved_model')
    else:
        model = PolicyGradientAgent(datum_size, num_rand_stocks, past_num)


    # training
    for i in range(NUM_EPOCHS):
        print(f'\n======================== EPOCH {i+1} of {NUM_EPOCHS} ========================')
        # start_day = randint(0, num_days - model.past_num - model.batch_size)  # TODO: inefficient usage of data?
        # sample = train_data[:, start_day:, :]
        episode_max_days = 200
        epoch_loss, rewards_list = train(train_data, model, x_tickers, RANDOMIZE, num_rand_stocks=num_rand_stocks, episode_max_days=episode_max_days)
        print(f"Avg Loss for epoch {i + 1} is {tf.reduce_mean(epoch_loss)}")
        # visualize_linegraph(rewards_list)
        # we use validating to determine when our model is overfitting / for adjusting hyperparameters
        if VALIDATING:
            print("\n===== validating on the same stocks ... =====")
            test(test_data_same_ticks, model, x_tickers, RANDOMIZE, num_rand_stocks=num_rand_stocks)
    if SAVE:
        save_model(model, 'saved_model')

    # if RANDOMIZE, model is trained on num_rand_stocks number of stocks
    # if not, model is trained on len(train_tickers) == train_data.shape[0] number of stocks
    # thus, if RANDOMIZE and "test on same stocks", we randomly pick num_rand_stocks number of stocks to test

    # testing
    # test on same stocks
    print("\n\n----- Testing on same stocks, RANDOMIZE:{} -----".format(RANDOMIZE))
    test(test_data_same_ticks, model, x_tickers, RANDOMIZE, num_rand_stocks=num_rand_stocks)

    # test on different stocks
    # we don't need to randomize test_data_diff_ticks when testing on different stocks
    print("\n\n----- Testing on different stocks -----")
    test(test_data_diff_ticks, model, y_tickers, False)

    print("END")


if __name__ == '__main__':
    main()
