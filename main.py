import numpy as np
import tensorflow as tf
from agent import PolicyGradientAgent
from stock_env import *


def train(train_data, model):
    """
    the train function, train the model for an entire epoch

    :param train_data: the preprocessed training data, of shape [num_stocks, state_size, num_days]
    :param model: the RL agent

    :return to be decided
    """
    batch_size = model.batch_size
    num_days = train_data.shape[2]
    # TODO: 0) batch the train_data, and for each batch:
    # TODO: 1) generate an episode, put it in the memory buffer (use generate_episode() from stock_env.py)
    # TODO: 2) sample a batch of (state, action, reward) from the memory buffer
    # TODO: 3) compute the discounted rewards
    # TODO: 4) compute the loss, run back prop on the model
    for batch in range(0, int(num_days / batch_size)):
        print("Training batch #{}".format(batch))
        start = batch * batch_size
        end = start + batch_size
        batch_input = train_data[:, :, start:end]
        with tf.GradientTape() as tape:
            states, actions, rewards = generate_episode(batch_input, model)
            discounted_rewards = discount(rewards)
            model.buffer.append((states, actions, discounted_rewards))
            repl_states, repl_actions, repl_discounted_rewards = model.experience_replay
            
            repl_states = tf.convert_to_tensor(repl_states)
            repl_actions = tf.convert_to_tensor(repl_actions)
            repl_discounted_rewards = tf.convert_to_tensor(repl_discounted_rewards)
            
            model_loss = model.loss(repl_states, repl_actions, repl_discounted_rewards)
        
        gradients = tape.gradient(los, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
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
