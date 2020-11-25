import numpy as np
import tensorflow as tf
from agent import PolicyGradientAgent
from stock_env import *
from visual_helpers import *


def train(train_data, model):
    """
    the train function, train the model for an entire epoch

    :param train_data: the preprocessed training data, of shape [num_stocks, num_days, datum_size]
    :param model: the RL agent

    :return to be decided
    """
    batch_size = model.batch_size
    num_days = train_data.shape[1]
    # TODO: 0) batch the train_data, and for each batch:
    # TODO: 1) generate an episode, put it in the memory buffer (use generate_episode() from stock_env.py)
    # TODO: 2) sample a batch of (state, action, reward) from the memory buffer
    # TODO: 3) compute the discounted rewards
    # TODO: 4) compute the loss, run back prop on the model
    for batch in range(0, int(num_days / batch_size)):
        print("Training batch #{}".format(batch))
        start = batch * batch_size
        end = start + batch_size
        batch_input = train_data[:, start:end, :]
        env = StockEnv(batch_input)
        with tf.GradientTape() as tape:
            states, actions, rewards, _ = env.generate_episode(model)
            discounted_rewards = discount(rewards)
            model.remember(states, actions, discounted_rewards)
            # repl_states, repl_actions, repl_discounted_rewards = model.experience_replay()
            repl_states, repl_actions, repl_discounted_rewards = states, actions, discounted_rewards
            # TODO: remove this after experience replay is finished
            # repl_states = tf.convert_to_tensor(repl_states)
            # repl_actions = tf.convert_to_tensor(repl_actions)
            # repl_discounted_rewards = tf.convert_to_tensor(repl_discounted_rewards)
            
            model_loss = model.loss(repl_states, repl_actions, repl_discounted_rewards)
        
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

    batch_size = model.batch_size
    num_days = test_data.shape[1]
    total_rewards = []

    batches = int(num_days / batch_size)
    portfolio_across_batches = np.zeros((model.num_stocks + 1, int(batches / 10)))
    for batch in range(0, batches):
        print("Training batch #{}".format(batch))
        start = batch * batch_size
        end = start + batch_size
        batch_input = test_data[:, start:end, :]
        env = StockEnv(batch_input)
        states, actions, rewards, portfolio_cash = env.generate_episode(model)
        total_rewards.append(rewards[-1]) #reward at end of batch

        if batch % 10 == 0:
            portfolio_across_batches[:, int(batch / 10)] = portfolio_cash

    visualize_portfolio(portfolio_across_batches, tickers)
    visualize_linegraph(total_rewards)
    print(f'final cash: {total_rewards[:-1]}')

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

    test_data, train_data, tickers = get_data()
    num_stocks, num_days, datum_size = test_data.shape
    past_num = 50

    model = PolicyGradientAgent(datum_size, num_stocks, past_num)

    train(train_data, model)
    test(test_data, model, tickers)
    print("END")

if __name__ == '__main__':
    main()
