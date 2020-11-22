def generate_episode(states, model):
    """
    generate an episode of experience based on the model's current policy and given states
    stops generating when episode_length > num_states, or the agent is broke (check using model.capital)
    since we are doing day trading, num_states = num_days (each day is a state)
    this assumes that the agent's actions have no impact on the stock environment.
    reminder: a state is (open, high, low, close, adjusted close, volume)

    :param states: a bunch of states, of size [num_stocks, state_size, num_states]
    :param model: the RL agent, which contains a policy of which actions to take

    :return tuple of lists (states, actions, rewards), where each list is of episode_length.
            episode_length is not necessarily the same as num_states, because the agent might be broke and
            terminate the episode early
    """
    # TODO: iterate through all the states
    # TODO: in each state, take an action according to the agent's policy, gather the reward, step into next state


def discount(rewards, discount_factor=.99):
    """
    Takes in a list of rewards for each time step in an episode, and returns a list of the discounted rewards
    for each time step.

    :param rewards: List of rewards from an episode [r_{t1},r_{t2},...]
    :param discount_factor: Gamma discounting factor to use, defaults to .99
    :returns: discounted_rewards: list containing the discounted rewards for each time step in the original rewards list
    """
    discounted_reward = [0] * len(rewards)
    # using dynamic programming to fill in the
    for i in range(len(discounted_reward)-1, -1, -1):  # iterate through indices back wards
        if (i + 1) < len(discounted_reward):
            discounted_reward[i] = rewards[i] + discount_factor * discounted_reward[i+1]
        else:
            discounted_reward[i] = rewards[i]
    return discounted_reward
