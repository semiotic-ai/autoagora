# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from asyncio import run
from typing import Type

import pytest

from environments.environment_factory import EnvironmentFactory
from environments.simulated_subgraph import (
    NoisyCyclicQueriesSubgraph,
    NoisyQueriesSubgraph,
    SimulatedSubgraph,
)


class TestSimulatedSubgraph:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "env_class", [NoisyQueriesSubgraph, NoisyCyclicQueriesSubgraph]
    )
    @pytest.mark.parametrize("number_of_steps", [1, 582])
    def test_step_reset(self, env_class: Type[SimulatedSubgraph], number_of_steps: int):
        """Tests if step() and reset() are working properly."""
        # Instantiate env.
        env = env_class()

        # Make n steps.
        env.step(number_of_steps=number_of_steps)
        assert env._step == number_of_steps

        # Reset.
        env.reset()
        assert env._step == 0

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "env_class", [NoisyQueriesSubgraph, NoisyCyclicQueriesSubgraph]
    )
    @pytest.mark.parametrize("cost_multiplier", [1e-6, 37e-2, 10234])
    def test_cost_multiplier(
        self, env_class: Type[SimulatedSubgraph], cost_multiplier: float
    ):
        """Tests if cost_multiplier set/get methods are working properly."""
        # Instantiate env.
        env = env_class()

        # Set cost multiplier.
        run(env.set_cost_multiplier(cost_multiplier))
        assert env.get_cost_multiplier() == cost_multiplier


class TestNoisyQueriesSubgraph:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "cost_multiplier, target_qps",
        [
            [0.0, 0.9999996940977731],
            [1e-6, 0.9933071490757152],
            [13e-7, 0.8807970779778824],
            [5e-6, 6.661338147750939e-16],
            [1e-3, 0.0],
        ],
    )
    def test_noiseless_queries_per_second(
        self, cost_multiplier: float, target_qps: float
    ):
        """Tests if queries_per_second() is working properly - with noise turned off"""
        # Instantiate env.
        env = NoisyQueriesSubgraph(noise=False)

        # Set cost multiplier.
        run(env.set_cost_multiplier(cost_multiplier))

        # Get queries per second.
        assert run(env.queries_per_second()) == target_qps

    def test_noise_queries_per_second(self):
        """Tests if noise is working."""
        # Instantiate env.
        env = NoisyQueriesSubgraph()

        # Set cost multiplier.
        run(env.set_cost_multiplier(1e-6))

        qps1 = run(env.queries_per_second())
        qps2 = run(env.queries_per_second())
        assert qps1 != qps2


class TestNoisyCyclicQueriesSubgraph:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "cost_multiplier, number_of_steps, target_qps",
        [
            [1e-6, 10, 0.5],
            [13e-7, 10, 0.047425873177566635],
            [1e-6, 1010, 0.9933071490757152],
            [13e-7, 1010, 0.8807970779778824],
        ],
    )
    def test_noisless_queries_per_second(
        self, cost_multiplier: float, number_of_steps: int, target_qps: float
    ):
        """Tests if queries_per_second() is working properly - with noise turned off"""
        # Instantiate env.
        env = NoisyCyclicQueriesSubgraph(noise=False)

        # Make n steps.
        env.step(number_of_steps=number_of_steps)

        # Set cost multiplier.
        run(env.set_cost_multiplier(cost_multiplier))

        assert run(env.queries_per_second()) == target_qps

    def test_noise_queries_per_second(self):
        """Tests if noise is working."""
        # Instantiate env.
        env = NoisyCyclicQueriesSubgraph()

        # Set cost multiplier.
        run(env.set_cost_multiplier(1e-6))

        qps1 = run(env.queries_per_second())
        qps2 = run(env.queries_per_second())
        assert qps1 != qps2


@pytest.mark.skipif(
    not importlib.util.find_spec("autoagora_isa"),
    reason=f'Optional package "autoagora-isa" is not installed.',
)
class TestIsaSubgraph:
    @pytest.mark.unit
    def test(self):
        from autoagora_isa.isa import IsaSubgraph  # type: ignore

        env = EnvironmentFactory(environment_type_name="isa")
        assert isinstance(env, SimulatedSubgraph)  # Makes Pyright happy when isa absent
        assert isinstance(env, IsaSubgraph)

        run(env.set_cost_multiplier(cost_multiplier=1e-6, agent_id=0))
        run(env.set_cost_multiplier(cost_multiplier=1e-6, agent_id=1))

        env.step()

        agent0_qps = run(env.queries_per_second(agent_id=0))
        agent1_qps = run(env.queries_per_second(agent_id=1))

        print(f"{agent0_qps=}")
        print(f"{agent1_qps=}")

        assert agent0_qps == pytest.approx(0.5, rel=0.3)
        assert agent1_qps == pytest.approx(0.5, rel=0.3)
