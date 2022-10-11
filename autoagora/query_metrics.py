# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import re
from typing import Dict

import aiohttp
import backoff
import configargparse

from autoagora.config import args

argsparser = configargparse.get_argument_parser()
argsparser.add_argument(
    "--indexer-service-metrics-endpoint",
    env_var="INDEXER_SERVICE_METRICS_ENDPOINT",
    required=True,
    help="HTTP endpoint for the indexer-service metrics.",
)


@backoff.on_exception(
    backoff.expo, aiohttp.ClientError, max_time=30, logger=logging.root
)
async def indexer_service_metrics() -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(args.indexer_service_metrics_endpoint) as response:
            assert response.status == 200, f"{response.status=}"
            result = await response.text()
    return result


@backoff.on_exception(
    backoff.expo, aiohttp.ClientError, max_time=30, logger=logging.root
)
async def query_counts() -> Dict[str, int]:
    async with aiohttp.ClientSession() as session:
        async with session.get(args.indexer_service_metrics_endpoint) as response:
            assert response.status == 200

            results = re.findall(
                r'indexer_service_queries_ok{deployment="(Qm[a-zA-Z0-9]{44})"} ([0-9]*)',
                await indexer_service_metrics(),
            )

    results = {subgraph: int(value) for subgraph, value in results}
    logging.debug("Query counts: %s", results)

    return results


@backoff.on_exception(
    backoff.expo, aiohttp.ClientError, max_time=30, logger=logging.root
)
async def subgraph_query_count(subgraph: str) -> int:
    async with aiohttp.ClientSession() as session:
        async with session.get(args.indexer_service_metrics_endpoint) as response:
            assert response.status == 200

            results = re.findall(
                r'indexer_service_queries_ok{{deployment="{subgraph}"}} ([0-9]*)'.format(
                    subgraph=subgraph
                ),
                await indexer_service_metrics(),
            )

    if len(results) == 0:
        # The subgraph query count will not be in the metric if it hasn't received any
        # queries.
        return 0
    if len(results) == 1:
        return int(results[0])
    else:
        raise RuntimeError(f"More than one matching metric entry for {subgraph}.")


if __name__ == "__main__":
    res = asyncio.run(
        subgraph_query_count("Qmaz1R8vcv9v3gUfksqiS9JUz7K9G8S5By3JYn8kTiiP5K")
    )
    print(res)
