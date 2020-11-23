import tensorflow as tf
import numpy as np
from preprocess import get_data

class StockEnv():
    def __init__(self,
                 is_testing=False,
                 initial_cash=1000,
                 buy_sell_amt=100,
                 exit_threshold=0,
                 inflation_annual=0.02,
                 interest_annual=0.03,
                 borrow_interest_annual=0.03,
                 transaction_penalty=0.0001):
        """
        Initializes a stock environment. The environment handles penalties resulting from inflation
        and borrowing. It ensures that the episode will exit when total cash value of assets < exit_threshold.

        Args:
            is_testing (boolean)
            initial_cash (number)
            buy_sell_amt (number): cash amount to buy or sell
            exit_threshold (number): minimum total cash value of assets
            inflation_annual (number): annual inflation rate
            interest_annual (number): annual interest rate
            borrow_interest_annual (number): annual stock loan fee
            transaction_penalty (number): transaction fee percentage
        """

        days_per_year = 261  # number of trading days per year
        self.initial_cash = initial_cash
        self.buy_sell_amt = buy_sell_amt
        self.exit_threshold = exit_threshold  # minimum total cash value of assets
        self.inflation = inflation_annual / days_per_year  # daily penalty on all assets
        self.interest = interest_annual / days_per_year  # daily penalty on borrowed cash
        self.borrow_interest = borrow_interest_annual / days_per_year  # daily penalty on borrowed stocks
        self.transaction_penalty = transaction_penalty

        data = get_data()  # pricing data
        self.pricing_data = data[1] if is_testing else data[0]

    def generate_episode_enumerated(self, model):
        """
        generate an episode of experience based on the model's current policy
        episode stops when the end of the pricing data is reached, or the agent is broke
        the environment handles recalculation of cash on hand, cash value of stocks, and total cash value of assets

        :param model: the RL agent, which contains a policy of which actions to take

        :return tuple of lists (states, actions, rewards), where each list is of episode_length. Note that
                episode_length is not necessarily the same as num_states, because the agent might be broke and
                terminate the episode early.

                assumes model.call outputs an array probabilities of shape (3^num_stocks, )
                containing the probability of each of the 3^num_stocks possible actions

                To derive an action, we sample a number between 0 and 3^num_stocks-1 from probabilities arrays
                We convert this number to base 3 array representation of 0s, 1s and 2s. Denote this string actions.
                E.g. sample from probabilities to get 10 -> actions = [0, 0, 1, 0, 1]
        """
        states = []
        actions = []
        rewards = []

        past_num = model.past_num
        num_stocks = model.num_stocks

        max_timestep = tf.shape(self.pricing_data)[1]
        timestep = past_num  # we start on day number <past_num>
        portfolio_cash = [0] * (num_stocks + 1)  # cash value of each asset
        portfolio_cash[num_stocks] = self.initial_cash  # cash on hand
        portfolio_shares = [0] * num_stocks  # shares of each stock owned
        total_cash_value = self.initial_cash

        # ================ GENERATION ================
        while timestep < max_timestep:  # we've reached the end of pricing_data

            sliced_price_history = self.pricing_data[:, timestep -
                                                     past_num:timestep, :]
            closing_prices = np.reshape(sliced_price_history[:, past_num, 3],
                                        (-1, ))

            # recalculate portfolio_cash based on new prices
            portfolio_cash[:-1] = portfolio_shares * closing_prices

            action = []  # joint action across all stocks
            transactions = 0  # number of buys and sells
            state = (sliced_price_history, portfolio_cash)
            probabilities = model.call([state]).numpy()[0]  #probabilities has shape 3**(num_stocks)

            # sample actions
            subaction_enumerated = np.random.choice(3**(num_stocks) - 1, 1, p=probabilities)
            action = numberToBase(subaction_enumerated, num_stocks= model.num_stocks)
            for i in range(len(action)):
                # 0=hold 1=buy 2=sell
                if action[i] == 1:  # buy
                    portfolio_cash[num_stocks] -= self.buy_sell_amt
                    portfolio_cash[i] += self.buy_sell_amt
                    portfolio_shares[i] += (self.buy_sell_amt /
                                            closing_prices[i])
                    transactions += 1
                elif action[i] == 2:  # sell
                    portfolio_cash[num_stocks] += self.buy_sell_amt
                    portfolio_cash[i] -= self.buy_sell_amt
                    portfolio_shares[i] -= (self.buy_sell_amt /
                                            closing_prices[i])
                    transactions += 1

            # transaction fees
            portfolio_cash[num_stocks] -= (transactions * self.buy_sell_amt *
                                           self.transaction_penalty)
            # inflation
            portfolio_cash *= 1 - self.inflation
            # recalculate total_cash_value
            total_cash_value = np.sum(portfolio_cash)

            # negative stock or negative cash
            for i in range(num_stocks+1):
                if portfolio_cash[i] < 0:
                    break
            time_reward = timestep - past_num

            states.append(state)
            actions.append(action)
            rewards.append(total_cash_value + time_reward)
            timestep += 1

        return states, actions, rewards

    def generate_episode(self, model):
        """
        generate an episode of experience based on the model's current policy
        episode stops when the end of the pricing data is reached, or the agent is broke
        the environment handles recalculation of cash on hand, cash value of stocks, and total cash value of assets

        :param model: the RL agent, which contains a policy of which actions to take

        :return tuple of lists (states, actions, rewards), where each list is of episode_length. Note that
                episode_length is not necessarily the same as num_states, because the agent might be broke and
                terminate the episode early.

                Each state is a tuple (<price_history>, <portfolio>) where price_history is of shape
                (model.num_stocks, model.past_num, datum_size), and portfolio is of shape (model.num_stocks+1,)
                where portfolio[model.num_stocks] = current cash on hand. Each element of portfolio in indices
                0 to model.num_stocks-1 is the cash value invested in the corresponding asset.

                Each action is of shape (model.num_stocks,) containing values of 0 (hold), 1 (buy), or 2 (sell),
                corresponding to the action taken for a stock.

                Each reward is a scalar representing the reward (i.e. total cash value of the portfolio).
        """
        states = []
        actions = []
        rewards = []

        past_num = model.past_num
        num_stocks = model.num_stocks

        timestep = past_num  # we start on day number <past_num>
        max_timestep = tf.shape(self.pricing_data)[1]
        portfolio_cash = [0] * (num_stocks + 1)  # cash value of each asset
        portfolio_cash[num_stocks] = self.initial_cash  # cash on hand
        portfolio_shares = [0] * num_stocks  # shares of each stock owned
        total_cash_value = self.initial_cash

        # ================ GENERATION ================
        while total_cash_value > self.exit_threshold:
            if timestep > max_timestep:  # we've reached the end of pricing_data
                break

            sliced_price_history = self.pricing_data[:, timestep -
                                                     past_num:timestep, :]
            closing_prices = np.reshape(sliced_price_history[:, past_num, 3],
                                        (-1, ))

            # recalculate portfolio_cash based on new prices
            portfolio_cash[:-1] = portfolio_shares * closing_prices

            action = []  # joint action across all stocks
            transactions = 0  # number of buys and sells
            state = (sliced_price_history, portfolio_cash)
            probabilities = model.call([state]).numpy()[0]  # batch_sz=1

            # sample actions
            for i in range(num_stocks):
                # 0=hold 1=buy 2=sell
                subaction = np.random.choice(3, 1, p=probabilities[i])[0]
                action.append(subaction)
                if subaction == 1:  # buy
                    portfolio_cash[num_stocks] -= self.buy_sell_amt
                    portfolio_cash[i] += self.buy_sell_amt
                    portfolio_shares[i] += (self.buy_sell_amt /
                                            closing_prices[i])
                    transactions += 1
                elif subaction == 2:  # sell
                    portfolio_cash[num_stocks] += self.buy_sell_amt
                    portfolio_cash[i] -= self.buy_sell_amt
                    portfolio_shares[i] -= (self.buy_sell_amt /
                                            closing_prices[i])
                    transactions += 1

            # transaction fees
            portfolio_cash[num_stocks] -= (transactions * self.buy_sell_amt *
                                           self.transaction_penalty)
            # borrowing stocks
            for i in range(num_stocks):
                if portfolio_cash[i] < 0:
                    portfolio_cash[num_stocks] += (portfolio_cash[i] *
                                                   self.borrow_interest)
            # borrowing cash
            if portfolio_cash[num_stocks] < 0:
                portfolio_cash[num_stocks] *= 1 + self.interest
            # inflation
            portfolio_cash *= 1 - self.inflation
            # recalculate total_cash_value
            total_cash_value = np.sum(portfolio_cash)

            states.append(state)
            actions.append(action)
            rewards.append(total_cash_value)
            timestep += 1

        return states, actions, rewards

def numberToBase(n, num_stocks = 5, b = 3):
    """
    convert integer n to base 3 representation as an array
    e.g. arr = numberToBase(10, num_stocks = 5, b = 3) = [0, 0, 1, 0, 1]
         arr = numberToBase(18, num_stocks = 5, b = 3) = [0, 0, 2, 0, 0]
         arr[i] indicates to hold = 0, buy = 1, or sell = 2
    """
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    base3_array = digits[::-1]

    actions = np.zeros(num_stocks)
    actions[num_stocks - len(digits[::-1]):] = digits[::-1]
    return actions

def discount(rewards, discount_factor=.99):
    """
    Takes in a list of rewards for each time step in an episode, and returns a list of the discounted rewards
    for each time step.

    :param rewards: List of rewards from an episode [r_{t1},r_{t2},...]
    :param discount_factor: Gamma discounting factor to use, defaults to .99
    :returns: discounted_rewards: list containing the discounted rewards for each time step in the original rewards list
    """
    length = len(rewards)
    discounted = np.zeros((length, ))
    accum = 0
    for i in range(length):
        accum = accum * discount_factor + rewards[i]
        discounted[length - i - 1] = accum
    return discounted