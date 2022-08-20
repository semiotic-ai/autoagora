# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod

from agents.mixin import ABCMixin


class Policy(ABCMixin):
    """Abstract policy class defining its elementary interface"""

    @abstractmethod
    def update_policy(self):
        """Abstract method for updating the agent's policy"""
        pass

    @abstractmethod
    def add_reward(self, reward):
        """Abstract method for adding reward to the buffer.

        Args:
            reward: reward to be adde.
        """
        pass


class NoUpdatePolicy(Policy):
    """Policy without any update."""

    def add_reward(self, reward):
        """Adds reward (empty function).

        Args:
            reward: reward to be added.
        """
        pass

    def update_policy(self):
        """Updates agent policy (empty function)."""
        pass


class ExperienceBufferPolicy(Policy):
    """Abstract policy class that stores and manages its own experience reply buffer with past actions and rewards.

    Args:
        buffer_max_size: (DEFAULT: 10) indicates the maximum size of buffer. If buffer_max_size>0, then the buffer will be truncated to this size.
    """

    def __init__(
        self,
        buffer_max_size: int = 10,
    ):
        # Call parent constructors.
        Policy.__init__(self)

        # Experience reply buffer.
        self.buffer_max_size = buffer_max_size
        self.action_buffer = []
        self.reward_buffer = []

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
