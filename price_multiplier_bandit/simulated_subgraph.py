# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from math import exp

import numpy as np

from price_multiplier_bandit.environment import Environment


class SimulatedSubgraph(Environment):
    """A simple abstract environment for simulating subgraph behavior.

    Args:
        cost_multiplier: (DEFAULT: 1e-6) Initial cost multiplier.
    """

    def __init__(self, cost_multiplier: float = 1e-6) -> None:
        # Set initial cost muptiplier.
        self._cost_multiplier = cost_multiplier
        # Reset step counter.
        self._step = 0

    def reset(self):
        """Resets step counter."""
        self._step = 0

    def step(self, number_of_steps: int = 1):
        """Executes step of the environment.

        Args:
            step_size: (DEFAULT: 1) Number of steps to perform.
        """
        self._step += number_of_steps

    @property
    def cost_multiplier(self):
        """Gets the cost multiplier."""
        return self._cost_multiplier

    @cost_multiplier.setter
    def cost_multiplier(self, cost_multiplier: float):
        """Sets the cost multiplier."""
        self._cost_multiplier = cost_multiplier

    async def set_cost_multiplier(self, cost_multiplier: float):
        """Sets the cost multiplier - async version."""
        self._cost_multiplier = cost_multiplier

    @staticmethod
    def sigmoid(x):
        """Static helper method, calculates sigmoid."""
        return 1 / (1 + exp(-x))

    async def observation(self):
        """Returns observation which in this case is number of queries per second."""
        return await self.queries_per_second()

    @abstractmethod
    async def queries_per_second(self):
        """Abstract method returning number of queries depending on the environment step."""
        pass

    def __str__(self):
        """
        Return:
            String describing the class and some of its main params.
        """
        return f"{self.__class__.__name__}(base_cost_multiplier={self.cost_multiplier})"


class NoisyQueriesSubgraph(SimulatedSubgraph):
    """A simple environment simulating subgraph with noisy target queries per second.

    Args:
        cost_multiplier: (DEFAULT: 1e-6) Initial cost multiplier.
        noise: (DEFAULT: True) If set, injects noise.
    """

    def __init__(self, cost_multiplier: float = 1e-6, noise: bool = True) -> None:
        # Set initial cost muptiplier.
        self._cost_multiplier = cost_multiplier
        # Reset step counter.
        self._step = 0
        # Set noise flag.
        self._noise = noise

    def __str__(self):
        """
        Return:
            String describing the class and highlighting of its main params.
        """
        return f"{self.__class__.__name__}(noise={self._noise})"

    async def queries_per_second(self):
        """Returns noisy number of queries depending on the environment step."""
        compress = 1e7
        shift = 1.5e8

        # Calculate basic value.
        queries_per_second = 1 - self.sigmoid(
            self._cost_multiplier * compress - shift / compress
        )

        # Add noise level.
        if self._noise:
            noise = np.random.normal() / 20
            queries_per_second *= 1 + noise

        return queries_per_second


class NoisyCyclicQueriesSubgraph(SimulatedSubgraph):
    """A simple environment simulating subgraph with noisy non-stationary target queries per second.

    Args:
        cost_multiplier: (DEFAULT: 1e-6) Initial cost multiplier.
        noise: (DEFAULT: True) If set, injects noise.
    """

    def __init__(
        self, cost_multiplier: float = 1e-6, cycle: int = 1000, noise: bool = True
    ) -> None:
        # Set initial cost muptiplier.
        self._cost_multiplier = cost_multiplier
        # Reset step counter.
        self._step = 0
        # Set noise flag.
        self._noise = noise
        # Remember cycle.
        self._cycle = cycle

    def __str__(self):
        """
        Return:
            String describing the class and highlighting of its main params.
        """
        return f"{self.__class__.__name__}(noise={self._noise}.cycle={self._cycle})"

    async def queries_per_second(self):
        """Returns noisy number of queries depending on the environment step."""
        compress = 1e7

        # Non-stationary, cyclic environment.
        if (self._step // self._cycle) % 2 == 1:
            shift = 1.5e8
        else:
            shift = 1e8

        # Calculate basic value.
        queries_per_second = 1 - self.sigmoid(
            self._cost_multiplier * compress - shift / compress
        )

        # Add noise level.
        if self._noise:
            noise = np.random.normal() / 20
            queries_per_second *= 1 + noise

        return queries_per_second
