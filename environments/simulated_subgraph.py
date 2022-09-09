# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from asyncio import run
from math import exp

import numpy as np

from environments.environment import Environment


class SimulatedSubgraph(Environment):
    """A simple abstract environment for simulating subgraph behavior."""

    def __init__(self) -> None:
        # Set initial cost muptiplier - for default (0th) agent.
        self._cost_multipliers = []

        # Reset step counter.
        self._step = 0

    def reset(self):
        """Resets step counter."""
        self._step = 0

    def step(self, number_of_steps: int = 1):
        """Executes step of the environment.

        Args:
            step_size (int, DEFAULT: 1): Number of steps to perform.
        """
        self._step += number_of_steps

    def get_cost_multiplier(self, agent_id: int = 0):
        """Gets the cost multiplier.

        Args:
            agent_id (int, DEFAULT: 0): Id of the agent (indexer).
        """
        return self._cost_multipliers[agent_id]

    async def set_cost_multiplier(self, cost_multiplier: float, agent_id: int = 0):
        """Sets the cost multiplier - async version.

        Args:
            agent_id (int, DEFAULT: 0): Id of the agent (indexer).
        """
        if agent_id >= len(self._cost_multipliers):
            # Append.
            self._cost_multipliers.append(cost_multiplier)
        else:
            # Override.
            self._cost_multipliers[agent_id] = cost_multiplier

    @staticmethod
    def sigmoid(x):
        """Static helper method, calculates sigmoid."""
        return 1 / (1 + exp(-x))

    async def observation(self, agent_id: int = 0):
        """Returns observation which in this case is number of queries per second.

        Args:
            agent_id (int, DEFAULT: 0): Id of the agent (indexer).
        """
        return await self.queries_per_second(agent_id)

    @abstractmethod
    async def queries_per_second(self, agent_id: int = 0):
        """Abstract method returning number of queries depending on the environment step.

        Args:
            agent_id (int, DEFAULT: 0): Id of the agent (indexer).
        """
        pass

    def __str__(self):
        """
        Return:
            String describing the class and some of its main params.
        """
        return f"{self.__class__.__name__}"

    async def generate_plot_data(
        self, min_x: float, max_x: float, num_points: int = 100, logspace: bool = False
    ):
        """Generates q/s for a given cost multiplier range.

        Args:
            min_x (float): Lower bound cost multiplier.
            max_x (float): Upper bound cost multiplier.
            num_points (int, optional): Number of points. Defaults to 100.

        Returns:
            ([x1, x2, ...], [y1, y2, ...]): Tuple of lists of x and y.
        """
        if logspace:
            x = np.logspace(np.log10(min_x), np.log10(max_x), num_points, base=10)
        else:
            x = np.linspace(min_x, max_x, num_points)
        y = []

        # ID of the "fake indexer" - add him to the end.
        visualizer_id = len(self._cost_multipliers)

        for val in x:
            # Set cost multiplier.
            await self.set_cost_multiplier(val, agent_id=visualizer_id)
            # Get observations, i.e. queries per second.
            y.append(await self.queries_per_second(agent_id=visualizer_id))

        # "Delete" the "fake indexer".
        self._cost_multipliers.pop()

        # Return x and y.
        return x, y


class NoisySimulatedSubgraph(SimulatedSubgraph):
    """A simple abstract environment for simulating noisy subgraph behavior.

    Args:
        noise: (DEFAULT: True) If set, injects noise.
    """

    def __init__(self, noise: bool = True) -> None:
        super().__init__()

        self._noise = noise


class NoisyQueriesSubgraph(NoisySimulatedSubgraph):
    """A simple environment simulating subgraph with noisy target queries per second.

    Args:
        noise: (DEFAULT: True) If set, injects noise.
    """

    def __init__(self, noise: bool = True) -> None:
        # Call parent class constructor.
        super().__init__()

        # Set noise flag.
        self._noise = noise

    def __str__(self):
        """
        Return:
            String describing the class and highlighting of its main params.
        """
        return f"{self.__class__.__name__}(noise={self._noise})"

    async def queries_per_second(self, agent_id: int = 0):
        """Returns noisy number of queries depending on the environment step.

        Args:
            agent_id (int, DEFAULT: 0): Id of the agent (indexer).
        """
        compress = 1e7
        shift = 1.5e8

        # Calculate basic value.
        queries_per_second = 1 - self.sigmoid(
            self._cost_multipliers[agent_id] * compress - shift / compress
        )

        # Add noise level.
        if self._noise:
            noise = np.random.normal() / 20
            queries_per_second *= 1 + noise

        return queries_per_second


class NoisyCyclicQueriesSubgraph(NoisySimulatedSubgraph):
    """A simple environment simulating subgraph with noisy non-stationary target queries per second.

    Args:
        cycle: (DEFAULT: 1000) Indicates how long a given cycle last.
        noise: (DEFAULT: True) If set, injects noise.
    """

    def __init__(self, cycle: int = 1000, noise: bool = True) -> None:
        # Call parent class constructor.
        super().__init__()

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

    async def queries_per_second(self, agent_id: int = 0):
        """Returns noisy number of queries depending on the environment step.

        Args:
            agent_id (int, DEFAULT: 0): Id of the agent (indexer).
        """
        compress = 1e7

        # Non-stationary, cyclic environment.
        if (self._step // self._cycle) % 2 == 1:
            shift = 1.5e8
        else:
            shift = 1e8

        # Calculate basic value.
        queries_per_second = 1 - self.sigmoid(
            self._cost_multipliers[agent_id] * compress - shift / compress
        )

        # Add noise level.
        if self._noise:
            noise = np.random.normal() / 20
            queries_per_second *= 1 + noise

        return queries_per_second


class NoisyCyclicZeroQueriesSubgraph(NoisySimulatedSubgraph):
    """A simple environment simulating subgraph with that for part of the cycle is not serving queries (queries per second = 0).

    Args:
        cycle: (DEFAULT: 1000) Indicates how long a given cycle last.
        noise: (DEFAULT: True) If set, injects noise (when queries > 0).
    """

    def __init__(self, cycle: int = 1000, noise: bool = True) -> None:
        # Call parent class constructor.
        super().__init__()

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

    async def queries_per_second(self, agent_id: int = 0):
        """Returns noisy number of queries depending on the environment step.

        Args:
            agent_id (int, DEFAULT: 0): Id of the agent (indexer).
        """
        compress = 1e7

        # Non-stationary, cyclic environment.
        if (self._step // self._cycle) % 2 == 1:
            queries_per_second = 0
        else:
            shift = 3e8

            # Calculate basic value.
            queries_per_second = 1 - self.sigmoid(
                self._cost_multipliers[agent_id] * compress - shift / compress
            )

            # Add noise level.
            if self._noise:
                noise = np.random.normal() / 20
                queries_per_second *= 1 + noise

        return queries_per_second


class NoisyDynamicQueriesSubgraph(NoisySimulatedSubgraph):
    """Environment simulating subgraph with variable number of queries changing at every cycle, with cycles where queries are not served at all (queries per second = 0).

    Args:
        noise: (DEFAULT: True) If set, injects noise (when queries > 0).
        cycle: (DEFAULT: 1000) Indicates how long a given cycle last.
    """

    def __init__(self, cycle: int = 1000, noise: bool = True) -> None:
        # Call parent class constructor.
        super().__init__()

        # Set noise flag.
        self._noise = noise
        # Remember cycle.
        self._cycle = cycle
        # Set initial queries per second.
        self.base_shift = self._sample_shift()

    def _sample_shift(self):
        """Sets the base q/s depending on the step.

        Returns:
            Base q/s.
        """
        # Sample multiplier.
        return np.random.ranf() * 5.0

    def __str__(self):
        """
        Return:
            String describing the class and highlighting of its main params.
        """
        return f"{self.__class__.__name__}(noise={self._noise}.cycle={self._cycle})"

    async def queries_per_second(self, agent_id: int = 0):
        """Returns noisy number of queries.

        Args:
            agent_id (int, DEFAULT: 0): Id of the agent (indexer).
        """

        # 20% chance for no queries in a given cycle.
        if self.base_shift < 1.0:
            return 0

        else:
            shift = 1e8
            compress = 1e7
            # Calculate basic q/s.
            queries_per_second = 1 - self.sigmoid(
                self._cost_multipliers[agent_id] * compress
                - self.base_shift * shift / compress
            )

        # Add noise level - at each step.
        if self._noise:
            noise = np.random.normal() / 20
            queries_per_second *= 1 + noise

        return queries_per_second

    def step(self, number_of_steps: int = 1):
        """Executes step of the environment.

        Args:
            step_size: (DEFAULT: 1) Number of steps to perform.
        """
        self._step += number_of_steps
        # Dynamic environment, changes every cycle.
        if (self._step % self._cycle) == 0:
            self.base_shift = self._sample_shift()
