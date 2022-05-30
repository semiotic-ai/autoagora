# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod


class Agent:
    """Abstract agent class defining agent's elementary interface"""

    @abstractmethod
    def get_action(self):
        """Abstract method returning agent's action."""
        pass

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
