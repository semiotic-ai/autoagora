# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from asyncio import sleep
from time import time
from typing import Optional

from autoagora.indexer_utils import get_cost_variables, set_cost_model
from autoagora.query_metrics import subgraph_query_count


class SubgraphWrapper:
    GATEWAY_DELAY = 60

    def __init__(self, subgraph) -> None:
        self.subgraph = subgraph
        self.last_change_time: Optional[float] = None

    async def set_cost_multiplier(self, cost_multiplier: float):
        cost_variables = await get_cost_variables(self.subgraph)
        cost_variables["GLOBAL_COST_MULTIPLIER"] = cost_multiplier
        await set_cost_model(self.subgraph, variables=cost_variables)
        self.last_change_time = time()

    async def queries_per_second(self, average_duration: float = 1):
        # Wait for the gateway to take our new costs into account
        if self.last_change_time is not None:
            time_since_last_change = time() - self.last_change_time
            if time_since_last_change < SubgraphWrapper.GATEWAY_DELAY:
                await sleep(SubgraphWrapper.GATEWAY_DELAY - time_since_last_change)

        timestamp_1 = time()
        query_count_1 = await subgraph_query_count(self.subgraph)

        await sleep(average_duration)

        timestamp_2 = time()
        query_count_2 = await subgraph_query_count(self.subgraph)

        queries_per_second = (query_count_2 - query_count_1) / (
            timestamp_2 - timestamp_1
        )

        return queries_per_second
