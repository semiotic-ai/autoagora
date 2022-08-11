# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from math import exp, log
from typing import Union, overload

import numpy as np
import scipy.stats as stats
import torch
from torch import distributions, nn

from agents.agent import Agent


class ContinuousActionBandit(Agent):
    """Abstract bandit class with continuous action space represented as a gausian.
    The agent internally stores and manages its own experience reply buffer with past actions and rewards.

    Args:
        learning_rate: learning rate.
        initial_mean: (DEFAULT: 0.0) initial mean.
        initial_logstddev: (DEFAULT: 0.4) initial (log) standard deviation.
        buffer_max_size: (DEFAULT: 30) indicates the maximum size of buffer. If buffer_max_size>0, then the buffer will be truncated to this size.

    """

    def __init__(
        self,
        learning_rate: float,
        initial_mean: float = 0.0,
        initial_logstddev: float = 0.4,
        buffer_max_size: int = 30,
    ):
        # Store init params.
        self._initial_mean = torch.Tensor([initial_mean])
        self._initial_logstddev = torch.Tensor([initial_logstddev])

        # Store policy params.
        self.mean = nn.parameter.Parameter(torch.Tensor([initial_mean]))
        self.logstddev = nn.parameter.Parameter(torch.Tensor([initial_logstddev]))

        # Experience reply buffer.
        self.buffer_max_size = buffer_max_size
        self.action_buffer = []
        self.reward_buffer = []

        # Initialize optimizer.
        self.optimizer = torch.optim.Adam([self.mean, self.logstddev], lr=learning_rate)
        self.learning_rate = learning_rate

    def __str__(self):
        """
        Return:
            String describing the class and highlighting of its main params.
        """
        return f"{self.__class__.__name__}(buffer_size={self.buffer_max_size}.learning_rate={self.learning_rate})"

    def distribution(self) -> torch.distributions.Normal:
        return distributions.Normal(self.mean, self.logstddev.clamp_max(5).exp())

    def get_bids(self):
        """Samples action from the action space, add it to action buffer and returns it.

        Return:
            Action sampled from the action space.
        """
        # Sample action from distribution.
        dist = self.distribution()
        action = dist.rsample().detach().item()
        assert isinstance(action, float)

        # Add action to buffer.
        self.action_buffer.append(action)

        return action

    def get_action(self):
        """Calls get_bids() and scale() to return scaled value."""
        bid = self.get_bids()
        scaled_bid = self.scale(bid)
        return scaled_bid

    @overload
    @staticmethod
    def scale(x: float) -> float:
        ...

    @overload
    @staticmethod
    def scale(x: torch.Tensor) -> torch.Tensor:
        ...

    @staticmethod
    def scale(x: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """Scales the value."""
        if isinstance(x, float):
            try:
                # print(f"x = {x}  => exp(x) * 1e-6 = {exp(x) * 1e-6}")
                return exp(x) * 1e-6
            except OverflowError:
                # print(f"!! OverflowError in exp(x) * 1e-6 for x = {x}!!")
                exit(-1)
        elif isinstance(x, torch.Tensor):
            return x.exp() * 1e-6
        else:
            raise TypeError(f"Invalid type '{type(x)}'")

    @overload
    @staticmethod
    def inv_scale(x: float) -> float:
        ...

    @overload
    @staticmethod
    def inv_scale(x: torch.Tensor) -> torch.Tensor:
        ...

    @staticmethod
    def inv_scale(x: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """Inverse operation to value scaling."""
        if isinstance(x, float):
            return log(x * 1e6)
        elif isinstance(x, torch.Tensor):
            return (x * 1e6).log()
        else:
            raise TypeError(f"Invalid type '{type(x)}'")

    def add_reward(self, reward):
        """Adds reward to the buffer.

        Args:
            reward: reward to be added.
        """
        self.reward_buffer.append(reward)

    def validate_experience_buffer(self):
        """Validates whether both buffers have the same size.

        Raises:
            ValueError if lengths of action and reward buffers are different.
        """
        if len(self.action_buffer) != len(self.reward_buffer):
            raise ValueError("Action and reward buffers need to be of the same size!")

    def is_experience_buffer_full(self):
        """
        Return:
            (True/False) informing whether the buffer is full.
        """
        # Check if buffer is full.
        if len(self.action_buffer) == self.buffer_max_size:
            return True
        else:
            return False

    def truncate_experience_buffer(self, buffer_max_size=None):
        """Truncates buffer size."""
        # Get max buffer size.
        if buffer_max_size is None:
            buffer_max_size = self.buffer_max_size

        # Truncate only if needed.
        if self.buffer_max_size > 0:
            while len(self.action_buffer) > self.buffer_max_size:
                self.action_buffer.pop(0)
                self.reward_buffer.pop(0)

    def clear_experience_buffer(self):
        """Clears the experience buffer."""
        self.action_buffer = []
        self.reward_buffer = []

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

        # Rescale x.
        # agent_min_x = self.inv_scale(min_x)
        # agent_max_x = self.inv_scale(max_x)

        # Prepare "scaled" and "unscaled" x.
        # agent_x = np.linspace(agent_min_x, agent_max_x, 200)
        # agent_x_scaled = [self.scale(x) for x in agent_x]

        agent_x_scaled = np.linspace(min_x, max_x, 200)
        agent_x = [self.inv_scale(x) for x in agent_x_scaled]

        # Get agent's PDF for "unscaled" x.
        policy_mean = self.mean.detach().numpy()
        policy_stddev = self.logstddev.exp().detach().numpy()
        policy_y = stats.norm.pdf(agent_x, policy_mean, policy_stddev) * policy_stddev

        # Get agent's init PDF for "unscaled" x.
        init_mean = self._initial_mean.detach().numpy()
        init_stddev = self._initial_logstddev.exp().detach().numpy()
        init_y = stats.norm.pdf(agent_x, init_mean, init_stddev) * init_stddev

        # Return x, y and iy.
        return agent_x_scaled, policy_y, init_y
