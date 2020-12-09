import tensorflow as tf
import numpy as np
from preprocess import get_data
from random import randint
from visual_helpers import visualize_portfolio, visualize_linegraph


class StockEnv():
    def __init__(self,
                 data,
                 tickers,
                 is_testing=False,
                 initial_cash=1000,
                 buy_sell_amt=100,
                 exit_threshold=0,
                 inflation_annual=0,
                 interest_annual=0,
                 borrow_interest_annual=0,
                 transaction_penalty=0.0001):
        """
        Initializes a stock environment. The environment handles penalties resulting from inflation
        and borrowing. It ensures that the episode will exit when total cash value of assets < exit_threshold.

        Args:
            data [num_stocks, num_days, state_size]: price data (set to None to input all available history data)
            tickers (num_stocks + 1, ): all reprocessed stocks, including cash
            is_testing (boolean)
            initial_cash (number)
            buy_sell_amt (number): cash amount to buy or sell
            exit_threshold (number): minimum total cash value of assets
            inflation_annual (number): annual inflation rate
            interest_annual (number): annual interest rate
            borrow_interest_annual (number): annual stock loan fee
            transaction_penalty (number): transaction fee percentage
        """
        self.is_testing = is_testing
        self.tickers = tickers
        days_per_year = 261  # number of trading days per year
        self.initial_cash = initial_cash
        self.buy_sell_amt = buy_sell_amt
        self.exit_threshold = exit_threshold
        self.inflation = inflation_annual / days_per_year  # daily penalty on all assets
        self.interest = interest_annual / days_per_year  # daily penalty on borrowed cash
        self.borrow_interest = borrow_interest_annual / days_per_year  # daily penalty on borrowed stocks
        self.transaction_penalty = transaction_penalty
        self.pricing_data = get_data(self.tickers) if data is None else data

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

        initial_timestep = past_num
        timestep = initial_timestep

        timestep_final = tf.shape(self.pricing_data)[1]
        portfolio_cash = np.zeros((num_stocks + 1,))  # cash value of each asset
        portfolio_cash[num_stocks] = self.initial_cash  # cash on hand
        portfolio_shares = np.zeros((num_stocks,))  # shares of each stock owned
        portfolio_shares = np.asarray(portfolio_shares)
        total_cash_value = self.initial_cash

        portfolio_cash_entire = np.zeros((num_stocks + 1, 1))

        first_step = True  # boolean variable used to create array for visualization
        # ================ GENERATION ================
        while timestep <= timestep_final and np.sum(portfolio_cash) > self.exit_threshold:
            sliced_price_history = self.pricing_data[:, timestep -
                                                        initial_timestep:timestep, :]
            closing_prices = np.reshape(sliced_price_history[:, -1, 3], (-1,))

            # recalculate portfolio_cash based on new prices
            portfolio_cash[:-1] = portfolio_shares * closing_prices

            action = []  # joint action across all stocks
            transactions = 0  # number of buys and sells
            state = tuple((sliced_price_history, portfolio_cash))

            probabilities = model.call([state])[0][0]  # batch_sz=1, take only the first arg
            probabilities = probabilities.numpy().reshape(num_stocks, 3)

            # sample actions
            for i in range(num_stocks):
                # 0=hold 1=buy 2=sell
                if np.isnan(probabilities[i][0]):
                    print("nan")
                    probabilities[i] = [1, 0, 0]
                # if self.is_testing:
                #     subaction = np.argmax(probabilities[i])
                # else:
                subaction = np.random.choice(3, 1, p=probabilities[i])[0]
                # print("probabilities for stock: {}".format(probabilities[i]))
                # print("subaction selected: {}".format(subaction))
                action.append(subaction)
                if subaction == 1:  # buy
                    portfolio_cash[num_stocks] -= self.buy_sell_amt
                    portfolio_cash[i] += self.buy_sell_amt
                    transactions += 1
                elif subaction == 2:  # sell
                    portfolio_cash[num_stocks] += self.buy_sell_amt
                    portfolio_cash[i] -= self.buy_sell_amt
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
            portfolio_cash = portfolio_cash * (1 - self.inflation)
            # recalculate portfolio_shares and total_cash_value
            portfolio_shares = portfolio_cash[:-1] / closing_prices
            total_cash_value = np.sum(portfolio_cash)

            states.append(state)
            actions.append(action)
            rewards.append(total_cash_value)
            
            # if self.is_testing:
            #     print("Timestep:", timestep)
            #     print("Closing Prices:", np.round(closing_prices, decimals=2))
            #     print("Action:", action)
            #     print("Portfolio:", np.round(total_cash_value, decimals=2), np.round(portfolio_cash, decimals=2))
            #     print("Portfolio shares:", np.round(portfolio_shares, decimals=2))

            timestep += 1

            # portfolio_cash_entire (num_stocks + 1, n): portfolio_cash across n time steps
            if first_step:
                portfolio_cash_entire[:, 0] = portfolio_cash
                first_step = False
            else:
                portfolio_cash_entire = np.hstack((portfolio_cash_entire, portfolio_cash.reshape((-1, 1))))
        # ================ END GENERATION ================

        print(f"Exit: timestep {timestep - past_num + 1} of {timestep_final - past_num + 1} with portfolio {np.round(portfolio_cash)}")

        # adjust rewards to be the difference between total portfolio values between two time steps
        delta_rewards = [rewards[i+1] - rewards[i] for i in range(len(rewards)-1)]
        delta_rewards = [rewards[0] - self.initial_cash] + delta_rewards  # first step reward
        # reward long episodes
        delta_rewards = [delta_rewards[i] + 100 * len(rewards) for i in range(len(rewards))]

        if self.is_testing:
            visualize_stride = int(portfolio_cash_entire.shape[1] / 10)
            visualize_portfolio(portfolio_cash_entire[:, ::visualize_stride], self.tickers)

            visualize_linegraph(rewards)

        return states, actions, rewards


def discount(rewards, discount_factor=.99):
    """
    Takes in a list of rewards for each time step in an episode, and returns a list of the discounted rewards
    for each time step.

    :param rewards: List of rewards from an episode [r_{t1},r_{t2},...]
    :param discount_factor: Gamma discounting factor to use, defaults to .99
    :returns: discounted_rewards: list containing the discounted rewards for each time step in the original rewards list
    """
    length = len(rewards)
    discounted = np.zeros((length,))
    accum = 0
    for i in range(length):
        accum = accum * discount_factor + rewards[length - i - 1]
        discounted[length - i - 1] = accum
    return discounted
