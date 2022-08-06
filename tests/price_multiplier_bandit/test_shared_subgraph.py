# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import random
from price_multiplier_bandit.shared_subgraph import NoisySharedSubgraph
from asyncio import run
import numpy as np
from typing import Sequence

class TestSharedSubgraph:
    @pytest.mark.unit
    @pytest.mark.parametrize("number_of_agents", [1, 2, 3, 4])
    @pytest.mark.parametrize("cost_multipliers", [[1e-6, 1e-6, 1e-6, 1e-6], [1.1e-6, 1.2e-6, 1.3e-6, 1.4e-6], [1e-7, 5e-5, 1e-9, 3e-6]])
    def test_sum_queries_close_multipliers(self, number_of_agents: int, cost_multipliers: Sequence[float]):
        
        env = NoisySharedSubgraph()

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
