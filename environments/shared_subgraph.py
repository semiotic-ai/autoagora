# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from asyncio import run
from math import exp

import numpy as np

from environments.simulated_subgraph import SimulatedSubgraph


class NoisySharedSubgraph(SimulatedSubgraph):
    """Environment simulating a shared subgraph where agents can compete over a query volume.

    Args:
        cost_multiplier_threshold: (DEFAULT: 1e-6) Cost multiplier threshold above which subgraph won't return any queries.
        noise: (DEFAULT: True) If set, injects noise (when queries > 0).
    """

    def __init__(
        self, cost_multiplier_threshold: float = 2e-6, noise: bool = True
    ) -> None:
        # Call parent class constructor.
        super().__init__()

        # Remember cost multiplier.
        self._cost_multiplier_threshold = cost_multiplier_threshold

        # Set noise flag.
        self._noise = noise

        # Set initial query volume.
        if self._noise:
            self._total_query_volume = 1 + np.random.normal() / 20
        else:
            self._total_query_volume = 1

    def __str__(self):
        """
        Return:
            String describing the class and highlighting of its main params.
        """
        return f"{self.__class__.__name__}(noise={self._noise})"

    @staticmethod
    def softmax(x):
        """Computes softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    async def queries_per_second(self, agent_id: int = 0):
        """Returns number of queries for a given agent.

        Args:
            agent_id (int, DEFAULT: 0): Id of the agent (indexer).
        """
        # Assign query number inversely proportional to cost multipliers.
        inv_cost_multipliers = [
            1 / cm if cm > 0 else 0 for cm in self._cost_multipliers
        ]

        # Only agents that set multiplier below the threshold will receive any queries.
        under_threshold = [
            1 if cm < self._cost_multiplier_threshold else 0
            for cm in self._cost_multipliers
        ]

        # Calculate the proportion for all agents.
        volume_proportion = self.softmax(
            [i * t for i, t in zip(inv_cost_multipliers, under_threshold)]
        )

        # Calculate qps for a given agent - and once again consider threshold.
        queries_per_second = (
            volume_proportion[agent_id]
            * under_threshold[agent_id]
            * self._total_query_volume
        )

        return queries_per_second

    def step(self, number_of_steps: int = 1):
        """Executes step of the environment.

        Args:
            step_size: (DEFAULT: 1) Number of steps to perform.
        """
        self._step += number_of_steps

        # Different noise => total query volume at every step.
        if self._noise:
            self._total_query_volume = 1 + np.random.normal() / 20

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
        x = [
            min_x,
            self._cost_multiplier_threshold,
            self._cost_multiplier_threshold,
            max_x,
        ]
        y = [self._total_query_volume, self._total_query_volume, 0, 0]

        # Return x and y.
        return x, y
