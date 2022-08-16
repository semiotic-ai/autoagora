# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from math import exp, log
import numpy as np
import scipy.stats as stats

from agents.agent import Agent

class RandomAgent(Agent):
    """Heuristic agent sampling an action from a continuous action space represented by a normal distribution.
    """

    def add_reward(self, reward):
        """Adds reward to the buffer (empty function).

        Args:
            reward: reward to be added.
        """
        pass

    def update_policy(self):
        """Updates agent policy (empty function)."""
        pass
