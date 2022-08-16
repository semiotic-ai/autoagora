# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

from agents.agent import Agent

class ContinuousActionBandit(Agent):
    """Abstract bandit class with continuous action space represented as a gausian.
    The agent internally stores and manages its own experience reply buffer with past actions and rewards.

    Args:
        learning_rate: learning rate.
        initial_mean: (DEFAULT: 0.0) initial mean in the original action (i.e. scaled bid) space.
        initial_stddev: (DEFAULT: 0.4) initial standard deviation in the original action (i.e. scaled bid) space.
        buffer_max_size: (DEFAULT: 10) indicates the maximum size of buffer. If buffer_max_size>0, then the buffer will be truncated to this size.

    """

    def __init__(
        self,
        learning_rate: float,
        initial_mean: float = 1e-6,
        initial_stddev: float = 1e-7,
        buffer_max_size: int = 10,
    ):
        # Call parent constructors.
        Agent.__init__(self)

        # Experience reply buffer.
        self.buffer_max_size = buffer_max_size
        self.action_buffer = []
        self.reward_buffer = []

        # Initialize optimizer.
        self.optimizer = torch.optim.Adam(params=self.params, lr=learning_rate)
        self.learning_rate = learning_rate

    def __str__(self):
        """
        Return:
            String describing the class and highlighting of its main params.
        """
        return f"{self.__class__.__name__}(buffer_size={self.buffer_max_size}.learning_rate={self.learning_rate})"


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
