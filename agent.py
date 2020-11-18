import numpy as np
import tensorflow as tf


class PolicyGradientAgent(tf.keras.models):
    def __init__(self, state_size, num_stocks):
        """
        this class inherits from tf.keras.models
        a general class of policy gradient RL agents using actor-critic

        :param state_size: the size of the state space, passed in from preprocessing
        :param num_stocks: the number of stocks the agent manages at once
        """
        super(PolicyGradientAgent, self).__init__()
        self.state_size = state_size
        self.num_stocks = num_stocks
        self.num_actions = 2 * self.num_stocks
        # TODO: init all the params, actor layers, critic layers, learning rate schedule, optimizer
        # TODO: init memory_buffer, buffer_size, batch_size
        # TODO: might use keras.Sequential. LOOK INTO THIS!

    def call(self, states):
        """
        performs the forward pass on a bunch of states, generate probabilities for taking each action

        :param states: a batch of input states [batch_sz * self.state_size]
        :return a tensor of size [batch_sz * self.num_actions], representing the action probability of taking each
                action
        """
        # TODO: perform the forward pass
        # TODO: do not use softmax, use an activation to squeeze results into [-1, 1]
        pass

    def value(self, states):
        """
        this is the critic, it estimates the value of being in each state using the value network

        :param states: a batch of input states [batch_sz * self.state_size]
        :return a tensor of size [batch_sz] representing the value of each inputted states
        """
        # TODO: call the value network, do not use softmax
        pass

    def loss(self, action_probs, actions_taken, discounted_reward):
        """
        computes the loss of a single forward pass for the agent

        :param action_probs:
        :param actions_taken:
        :param discounted_reward:
        """
        # TODO:
        pass

    def remember(self, states, actions, rewards):
        """

        """
        # TODO:
        pass

    def experience_replay(self):
        """

        """
        # TODO:
