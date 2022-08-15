# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from math import exp, log
import numpy as np
import scipy.stats as stats

from agents.agent import Agent

class RandomAgent(Agent):
    """Heuristic agent sampling an action from a continuous action space represented by a normal distribution.

    Args:
        initial_mean: (DEFAULT: 0.0) initial mean.
        initial_stddev: (DEFAULT: 1e-7) initial standard deviation.

    """

    def __init__(
        self,
        initial_mean: float = 0.0,
        initial_stddev: float = 1e-7,
    ):
        # Store init params.
        self.mean = initial_mean
        self.stddev = initial_stddev


    def get_bids(self):
        """Samples action from the action space, add it to action buffer and returns it.

        Return:
            Action sampled from the action space.
        """
        # Sample action from distribution.
        action = np.random.normal(loc=self.mean, scale=self.stddev)
        print(action)

        return action

    def get_action(self):
        """Calls get_bids() and scale() to return scaled value."""
        bid = self.get_bids()
        scaled_bid = self.scale(bid)
        return scaled_bid

    @staticmethod
    def scale(x: float) -> float:
        """Scales the value (no scaling!)"""
        return x

    @staticmethod
    def inv_scale(x: float) -> float:
        """Inverse operation to value scaling (no scaling!)"""
        return x

    def add_reward(self, reward):
        """Adds reward to the buffer (empty function).

        Args:
            reward: reward to be added.
        """
        pass

    def update_policy(self):
        """Updates agent policy (empty function)."""
        pass

    async def generate_plot_data(
        self, min_x: float, max_x: float, num_points: int = 200
    ):
        """Generates action distribution for a given cost multiplier range.

        Args:
            min_x (float): Lower bound cost multiplier.
            max_x (float): Upper bound cost multiplier.
            num_points (int, optional): Number of points. Defaults to 200.

        Returns:
            ([x1, x2, ...], [y1, y2, ...], [iy1, iy2, ...]): Triplet of lists of x, y (current policy PDF) and iy (init policy PDF).
        """
        agent_x = np.linspace(min_x, max_x, num_points)

        # Get agent's PDF for "unscaled" x.
        policy_mean = self.mean
        policy_stddev = self.stddev
        policy_y = stats.norm.pdf(x=agent_x, loc=policy_mean, scale=policy_stddev)
        # Scale y to max=0.5.
        policy_y /= max(policy_y) * 2

        # Get agent's init PDF for "unscaled" x.
        init_y = policy_y

        # Return x, y and iy.
        return agent_x, policy_y, init_y


class RandomScaledAgent(Agent):
    """Heuristic agent sampling an action from a continuous action space represented by a normal distribution.
    Action space is scaled.

    Args:
        initial_mean: (DEFAULT: 0.0) initial mean (actually initial SCALED mean)
        initial_stddev: (DEFAULT: 1e-7) initial standard deviation (actually initial SCALED std).

    """

    def __init__(
        self,
        initial_mean: float = 0.0,
        initial_stddev: float = 1e-7,
    ):
        # Store init params.
        self.mean = initial_mean
        self.stddev = initial_stddev

    def get_bids(self):
        """Samples action from the action space, add it to action buffer and returns it.

        Return:
            Action sampled from the action space.
        """
        # Sample action from distribution.
        action = np.random.normal(loc=self.mean, scale=self.stddev)
        print(action)

        return action

    def get_action(self):
        """Calls get_bids() and scale() to return scaled value."""
        bid = self.get_bids()
        scaled_bid = self.scale(bid)
        return scaled_bid

    @staticmethod
    def scale(x: float) -> float:
        """Scales the value."""
        return exp(x) * 1e-6

    @staticmethod
    def inv_scale(x: float) -> float:
        """Inverse operation to value scaling."""
        return log(x * 1e6)

    def add_reward(self, reward):
        """Adds reward to the buffer (empty function).

        Args:
            reward: reward to be added.
        """
        pass

    def update_policy(self):
        """Updates agent policy (empty function)."""
        pass

    async def generate_plot_data(
        self, min_x: float, max_x: float, num_points: int = 200
    ):
        """Generates action distribution for a given cost multiplier range.

        Args:
            min_x (float): Lower bound cost multiplier.
            max_x (float): Upper bound cost multiplier.
            num_points (int, optional): Number of points. Defaults to 200.

        Returns:
            ([x1, x2, ...], [y1, y2, ...], [iy1, iy2, ...]): Triplet of lists of x, y (current policy PDF) and iy (init policy PDF).
        """

        agent_x_scaled = np.linspace(min_x, max_x, 300)
        #agent_x = [self.inv_scale(x) for x in agent_x_scaled]

        # Get agent's PDF for "unscaled" x.
        policy_mean = self.mean
        policy_stddev = self.stddev
        policy_y = stats.norm.pdf(x=agent_x_scaled, loc=policy_mean, scale=policy_stddev)
        policy_y /= max(policy_y) * 2
        #policy_y = stats.norm.pdf(agent_x, policy_mean, policy_stddev) * policy_stddev

        print("policy_mean = ", policy_mean)
        print("policy_stddev = ", policy_stddev)
        #print("policy_y = ", policy_y)
        #print("agent_x_scaled = ", agent_x_scaled)
        # Get agent's init PDF for "unscaled" x.
        #init_y = policy_y

        # Return x, y and iy.
        return agent_x_scaled, policy_y, None #init_y


