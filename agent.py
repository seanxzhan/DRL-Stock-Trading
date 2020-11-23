import numpy as np
import tensorflow as tf
import random



class PolicyGradientAgent(tf.keras.Model):
    def __init__(self, datum_size, num_stocks, past_num, resume=False):
        """
        this class inherits from tf.keras.Model
        a general class of policy gradient RL agents using actor-critic
        assume this agent can manage num_stocks stocks at the same time, and the agent's actions have no impact on
        the stock environment. When performing each action, assume the agent can only sell/buy/hold one of the
        num_stocks stocks.

        (outdated, don't read, kept for documentation purpose)
        # the agent's action space is a tensor of size [3 * num_stocks], where indexes [3i, 3i+1, 3i+2] refers to the
        # i-th stock (zero-indexing). Each element of this tensor is in [0, 1] and all elements should sum up to 1 (this
        # tensor is a probability of taking each action). For the i-th stock, action[3i] is the probability of holding,
        # action[3i+1] is the probability of buying, action[3i+2] is the probability of selling.
        # To choose an action, we sample from this probability tensor and get an index k. if (k mod 3) == 0, then the
        # agent will do nothing. if (k mod 3) == 1, then the agent will buy (action[k] * self.max_buy) amount of the
        # corresponding stock. If (k mod 3) == 2, then the agent will sell (action[k] * self.max_sell) amount of the
        # corresponding stock.

        :param datum_size: the size of the state space, passed in from pre-processing
        :param num_stocks: the number of stocks the agent manages at once
        :param past_num: the past number of days to consider when making the current decision
        :param resume: True if creating the model from a previously trained model, False if training a new model from
                        random weights
        """
        super(PolicyGradientAgent, self).__init__()

        # stock market decision hyper params
        self.max_sell = 100  # the dollar amount the agent is willing to sell at most in one action
        self.max_buy = 100  # the dollar amount the agent is willing to buy at most in one action
        self.capital = 1000  # the initial amount of capital the agent holds
        self.past_num = past_num  # the number of days of historical financial data the agent will consider when
                                    # making the current decision.

        # RL agent params
        self.datum_size = datum_size
        self.num_stocks = num_stocks
        self.batch_size = 50
        self.buffer = []  # initialize the memory replay buffer
        self.buffer_size = 100  # maximum episodes the buffer can hold
        self.buffer_num_elt = 0  # the number of current elements in the buffer
        self.buffer_episode_lens = []  # a list of `episode length` of experience that were stored in the buffer
                                        # used to facilitate self.forget()
        self.num_actions = 3 * self.num_stocks
        self.actor_H1 = 24  # hidden layer output sizes
        self.actor_H2 = 24
        self.critic_H1 = 64
        self.critic_H2 = 16
        self.lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[30, 110, 220, 300],
                                                                                values=[0.01, 0.005, 0.003, 0.002,
                                                                                        0.001])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)

        # model layers
        self.resume = resume
        self.actor_gru_1 = tf.keras.layers.GRU(self.actor_H1, return_sequences=True, return_state=True)
        self.actor_dropout_1 = tf.keras.layers.Dropout(rate=0.1)
        self.actor_gru_2 = tf.keras.layers.GRU(self.actor_H2)
        self.actor_dropout_2 = tf.keras.layers.Dropout(rate=0.1)
        self.actor_dense = tf.keras.layers.Dense(self.num_actions)
        self.critic_dense_1 = tf.keras.layers.Dense(self.critic_H1, activation='relu')
        self.critic_dense_2 = tf.keras.layers.Dense(self.critic_H2, activation='relu')
        self.critic_dense_3 = tf.keras.layers.Dense(1)
        # attempt to use the keras.Sequential API
        # if resume:
        #     self.actor = tf.keras.models.load_model("saved_actor_model")
        #     self.critic = tf.keras.models.load_model("saved_critic_model")
        # else:
        #     # actor
        #     self.actor = tf.keras.models.Sequential()
        #     self.actor.add(tf.keras.layers.GRU(self.actor_H1))
        #     self.actor.add(tf.keras.layers.Dropout(rate=0.1))
        #     self.actor.add(tf.keras.layers.GRU(self.actor_H2))
        #     self.actor.add(tf.keras.layers.Dropout(rate=0.1))
        #     self.actor.add(tf.keras.layers.Dense(self.num_actions, activation='softmax'))
        #     self.actor.compile(optimizer=self.optimizer,
        #           loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False))
        #     # critic
        #     self.critic = tf.keras.models.Sequential()
        #     self.critic.add(tf.keras.layers.Concatenate())
        #     self.critic.add(tf.keras.layers.Dense(self.critic_H1, activation='relu'))
        #     self.critic.add(tf.keras.layers.Dense(self.critic_H2, activation='relu'))
        #     self.critic.add(tf.keras.layers.Dense(1))
        #     self.critic.compile(optimizer=self.optimizer, loss=tf.keras.losses.MSE)

    def call(self, states):
        """
        performs the forward pass on a bunch of states, generate probabilities for taking each action
        uses 2 GRU to summarize sequential stock data, and a dense layer to classify to actions

        :param states: a batch of state tuples (self.batch_sz, ), where each tuple is (<price_history>, <portfolio>)
                        where price_history is of shape (model.num_stocks, model.past_num, datum_size),
                        and portfolio is of shape (model.num_stocks+1,)
        :return a tensor of size (batch_sz, self.num_stock, 3), representing the action probability of taking each
                action. The tensor always sums to 1 on the last dimension (the prob of buying/selling/holding a
                particular stock on a particular day sums to one)
                a tensor that summarize the states tensor of shape [batch_sz * self.actor_H2]
        """
        # extract info from states
        price_history = [state[0] for state in states]  # [batch_sz, num_stocks, past_num, datum_size]
        price_history = tf.reshape(price_history, (-1, self.num_stocks * self.past_num, self.datum_size))  # TODO: this seems wrong
        portfolio = [state[1] for state in states]  # [batch_sz * (num_stock+1)]
        # pass through layers
        gru_1_out_whole_seq, _ = self.actor_dropout_1(self.actor_gru_1(price_history))  # (batch_sz, num_stock * past_num , actor_H1)
        gru_2_out = self.actor_dropout_2(self.actor_gru_2(gru_1_out_whole_seq))  # [batch_sz, actor_H2]
        past_and_current_info = tf.concat([gru_2_out, portfolio], axis=1)  # (batch_sz, actor_H2 + num_stock + 1)
        actor_out = self.actor_dense(past_and_current_info)  # (batch_sz * self.num_actions)
        # reshape the output and softmax
        actor_out = tf.reshape(actor_out, (-1, self.num_stocks, 3))
        action_probs = tf.nn.softmax(actor_out, axis=-1)  # TODO: not sure if this sums the last dim to 1
        return action_probs, past_and_current_info

    def value(self, states_summary):
        """
        this is the critic, it estimates the value of being in each state using the value network
        pass the summary of states info through 3 dense layers

        :param states_summary: a batch of summary of input states (self.batch_sz, self.num_actions), this is the second
                                output of the call function
        :return a tensor of size (batch_sz, 1) representing the value of each inputted states
        """
        hidden_1 = self.critic_dense_1(states_summary)  # [batch_sz * self.critic_H1]
        hidden_2 = self.critic_dense_2(hidden_1)  # [batch_sz * self.critic_H2]
        value_out = self.critic_dense_3(hidden_2)  # [batch_sz * 1]
        return value_out

    def loss(self, states, actions_taken, discounted_reward):
        """
        computes the loss of a single forward pass for the agent

        :param states: a batch of state tuples (self.batch_sz, ), where each tuple is (<price_history>, <portfolio>)
                        where price_history is of shape (model.num_stocks, model.past_num, datum_size),
                        and portfolio is of shape (model.num_stocks+1,)
        :param actions_taken: a batch of the sequence of actions that the agent actually took in the episode.(batch_sz,)
                                Each action is of shape (model.num_stocks,) containing values of
                                0 (hold), 1 (buy), or 2 (sell), corresponding to the action taken for a stock.
        :param discounted_reward: discounted rewards through the batch. (batch_sz, )
        :return a scalar loss of the whole batch
        """
        batch_sz = len(actions_taken)
        action_probs, states_summary = self.call(states)  # action_probs: (batch_sz, self.num_stock, 3)
        values = self.value(states_summary)  # (batch_sz, 1)
        values = tf.reshape(values, (-1))  # (batch_sz,)
        probs_of_action_taken = tf.zeros((batch_sz,))
        for b in range(batch_sz):
            probs_of_action_taken[b] = tf.reduce_prod(
                                        tf.gather_nd(action_probs[b],
                                                     list(zip(np.arange(self.num_stocks), actions_taken[b]))))
            # TODO: this could be very wrong...
        advantage = discounted_reward - values  # (batch_sz,)
        actor_loss = - tf.reduce_sum(tf.math.multiply(tf.math.log(probs_of_action_taken), tf.stop_gradient(advantage)))
        critic_loss = tf.reduce_sum(tf.math.square(advantage))
        loss = actor_loss + critic_loss  # scalar loss of the batch
        return loss

    # def actor_loss(self, action_probs, actions_taken, discounted_reward):
    #     """
    #     computes the loss of a single forward pass for the agent's actor module
    #
    #     :param action_probs: a batch of probabilities of the agent taking each aciton. [batch_sz * self.num_actions]
    #     :param actions_taken: the sequence of actions that the agent actually took in the episode. [batch_sz]
    #     :param discounted_reward: discounted rewards through the batch. [batch_sz]
    #     """
    #     pass
    #
    # def critic_loss(self, action_probs, actions_taken, discounted_reward):
    #     """
    #     computes the loss of a single forward pass for the agent's critic module
    #
    #     :param action_probs: a batch of probabilities of the agent taking each aciton. [batch_sz * self.num_actions]
    #     :param actions_taken: the sequence of actions that the agent actually took in the episode. [batch_sz]
    #     :param discounted_reward: discounted rewards through the batch. [batch_sz]
    #     """
    #     pass

    def remember(self, states, actions, discounted_rewards):
        """
        put an entire episode of states, actions, rewards into the memory-replay buffer
        call this function right after an entire episode of (s,a,r) is generated to remember the episode

        :param states: the sequence of states in an episode. [episode_len]
        :param actions: the sequence of actions in an episode. [episode_len]
        :param discounted_rewards: the sequence of rewards received in an episode. [episode_len]
        :return Nothing
        """
        if self.buffer_num_elt >= self.buffer_size:
            self.forget()
        episode_len = len(states)
        list_of_pairs = list(zip(states, actions, discounted_rewards))
        self.buffer += list_of_pairs
        self.buffer_episode_lens.append(episode_len)

    def forget(self):
        """
        get rid of the earliest episode in the memory buffer, because the experience is no longer useful.
        """
        earliest_episode_len = self.buffer_episode_lens.pop(0)  # number of elements to remove
        del self.buffer[:earliest_episode_len]  # remove the first n elements

    def experience_replay(self):
        """
        sample from the current memory buffer to get (state, action, discounted_reward) pairs and train on these
        experience replay will make the stock data more iid and stablize training
        This function is used to sample from self.buffer

        :return (states, actions_taken, discounted_reward), each of which is a list of length [batch_sz]
                if the buffer is large enough, else the length is [self.buffer_size]
        """
        if len(self.buffer) < self.batch_size:
            # not enough experience to sample from, just use all exp
            states, actions_taken, discounted_reward = list(zip(*self.buffer))  # unzip the buffer
            return states, actions_taken, discounted_reward
        else:
            # enough experience to sample from
            list_of_pairs = random.sample(self.buffer, self.batch_size)  # sample batch_size of samples from buffer
            states, actions_taken, discounted_reward = list(zip(*list_of_pairs))
            return states, actions_taken, discounted_reward

