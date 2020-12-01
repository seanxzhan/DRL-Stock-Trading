import tensorflow as tf
import numpy as np
from preprocess import get_data
from random import randint
from visual_helpers import visualize_portfolio, visualize_linegraph

class StockEnv():
    def __init__(self,
                 data,
                 all_tickers,
                 is_testing=False,
                 initial_cash=1000,
                 buy_sell_amt=100,
                 exit_threshold=0,
                 max_days= 100,
                 inflation_annual=0.02,
                 interest_annual=0.03,
                 borrow_interest_annual=0.03,
                 transaction_penalty=0.0001):
        """
        Initializes a stock environment. The environment handles penalties resulting from inflation
        and borrowing. It ensures that the episode will exit when total cash value of assets < exit_threshold.

        Args:
            data [num_stocks, num_days, state_size]: price data
            all_tickers (num_stocks + 1, ): all reprocessed stocks, including cash
            is_testing (boolean)
            initial_cash (number)
            buy_sell_amt (number): cash amount to buy or sell
            exit_threshold (number): minimum total cash value of assets
            max_days (number): maximum number of days to run
            inflation_annual (number): annual inflation rate
            interest_annual (number): annual interest rate
            borrow_interest_annual (number): annual stock loan fee
            transaction_penalty (number): transaction fee percentage
        """
        self.is_testing = is_testing
        days_per_year = 261  # number of trading days per year
        self.initial_cash = initial_cash
        self.buy_sell_amt = buy_sell_amt
        self.exit_threshold = exit_threshold
        self.max_days = max_days
        self.inflation = inflation_annual / days_per_year  # daily penalty on all assets
        self.interest = interest_annual / days_per_year  # daily penalty on borrowed cash
        self.borrow_interest = borrow_interest_annual / days_per_year  # daily penalty on borrowed stocks
        self.transaction_penalty = transaction_penalty

        #data = get_data()  # pricing data
        self.pricing_data = data
        self.all_tickers = all_tickers

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

        timestep_stop = tf.shape(self.pricing_data)[1] + 1 # if self.max_days is None else timestep + self.max_days
        portfolio_cash = [0] * (num_stocks + 1)  # cash value of each asset
        portfolio_cash[num_stocks] = self.initial_cash  # cash on hand
        portfolio_shares = [0] * num_stocks  # shares of each stock owned
        portfolio_shares = np.asarray(portfolio_shares)
        total_cash_value = self.initial_cash

        portfolio_cash_entire = np.zeros((num_stocks + 1, 1))

        first_step = True #boolean variable used to create array for visualization
        # ================ GENERATION ================
        while total_cash_value > self.exit_threshold:
            if timestep >= timestep_stop:  # we've reached the end of pricing_data
                break

            sliced_price_history = self.pricing_data[:, timestep -
                                                     initial_timestep:timestep, :]
            closing_prices = np.reshape(sliced_price_history[:, -1, 3],(-1,))

            # recalculate portfolio_cash based on new prices
            portfolio_cash[:-1] = portfolio_shares * closing_prices

            action = []  # joint action across all stocks
            transactions = 0  # number of buys and sells
            state = tuple((sliced_price_history, portfolio_cash))

            probabilities = model.call([state])[0]  # batch_sz=1
            probabilities = probabilities.numpy().reshape(num_stocks, 3)
            # print(probabilities)

            # sample actions
            for i in range(num_stocks):
                # 0=hold 1=buy 2=sell
                if np.isnan(probabilities[i][0]): #TODO: fix agent outputting nan probabilities
                    print("nan")
                    probabilities[i] = [1/3, 1/3, 1/3]
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
            portfolio_cash = np.asarray(portfolio_cash)*(1 - self.inflation)
            # recalculate total_cash_value
            total_cash_value = np.sum(portfolio_cash)

            states.append(state)
            actions.append(action)
            rewards.append(total_cash_value)
            timestep += 1

            #portfolio_cash_entire (num_stocks + 1, n): portfolio_cash across n time steps
            if first_step == True:
                portfolio_cash_entire[:, 0] = portfolio_cash
                first_step = False
            else:
                portfolio_cash_entire = np.hstack((portfolio_cash_entire, portfolio_cash.reshape((-1, 1))))

        if self.is_testing:
            print(portfolio_cash_entire)
            visualize_stride = int(portfolio_cash_entire.shape[1] / 10)
            visualize_portfolio(portfolio_cash_entire[:, ::visualize_stride], self.all_tickers)
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
    discounted = np.zeros((length, ))
    accum = 0
    for i in range(length):
        accum = accum * discount_factor + rewards[length - i - 1]
        discounted[length - i - 1] = accum
    return discounted
