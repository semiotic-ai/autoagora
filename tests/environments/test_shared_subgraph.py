# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from asyncio import run
from typing import Sequence

import numpy as np
import pytest

from environments.shared_subgraph import NoisySharedSubgraph


class TestSharedSubgraph:
    @pytest.mark.unit
    @pytest.mark.parametrize("number_of_agents", [1, 2, 3, 4])
    @pytest.mark.parametrize(
        "cost_multipliers",
        [
            [1e-7, 1e-7, 1e-7, 1e-7],
            [1.1e-7, 1.2e-7, 1.3e-7, 1.4e-7],
            [1e-7, 5e-5, 1e-9, 3e-6],
        ],
    )
    def test_query_volume(
        self, number_of_agents: int, cost_multipliers: Sequence[float]
    ):
        env = NoisySharedSubgraph(cost_multiplier_threshold=1e-6)

        # Run the environment for 3 steps
        for _ in range(3):
            queries_sum = 0

            # Set the cost multipliers for each of the agents
            for agent_id in range(number_of_agents):
                run(env.set_cost_multiplier(cost_multipliers[agent_id], agent_id))

            # Step the environment
            env.step()

            # Sum up the QPS of all the agents
            for agent_id in range(number_of_agents):
                qps = run(env.queries_per_second(agent_id))
                print(f"QPS agent {agent_id + 1}/{number_of_agents}: {qps}")
                queries_sum += run(env.queries_per_second(agent_id))

            total_query_volume = env._total_query_volume
            print(f"Environment's total QPS: {total_query_volume}")

            # Check that the agent's cumulative QPS is equal to the environment's total
            assert np.isclose(queries_sum, total_query_volume)

    @pytest.mark.unit
    @pytest.mark.parametrize("cost_multiplier_threshold", [1.234e-7, 1e-6, 4.126e-5])
    @pytest.mark.parametrize("agent_cost_multiplier", [2.345e-8, 1e-6, 6.234e-5])
    def test_no_queries_above_threshold(
        self, cost_multiplier_threshold: float, agent_cost_multiplier: float
    ):
        env = NoisySharedSubgraph(cost_multiplier_threshold=cost_multiplier_threshold)

        for _ in range(2):
            # Set the agent's cost multiplier
            run(env.set_cost_multiplier(agent_cost_multiplier))

            # Step the environment
            env.step()

            # Check that the agent's QPS is 0 is above the theshold, or the
            # environment's total query volume if under the threshold.
            agent_qps = run(env.queries_per_second())
            env_qps = env._total_query_volume
            if agent_cost_multiplier < cost_multiplier_threshold:
                assert np.isclose(agent_qps, env_qps)
            else:
                assert np.isclose(agent_qps, 0)
