# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import asyncio as aio
import logging
from dataclasses import dataclass
from typing import Dict, Optional

import psycopg_pool
from prometheus_async.aio.web import start_http_server

from autoagora.config import args, init_config
from autoagora.indexer_utils import get_allocated_subgraphs, set_cost_model
from autoagora.logs_db import LogsDB
from autoagora.model_builder import (
    apply_default_model,
    model_update_loop,
    mrq_model_update_loop,
)
from autoagora.price_multiplier import price_bandit_loop
from autoagora.query_metrics import (
    K8SServiceWatcherMetricsEndpoints,
    StaticMetricsEndpoints,
)
from autoagora.utils.constants import DEFAULT_AGORA_VARIABLES


@dataclass
class SubgraphUpdateLoops:
    bandit: Optional[aio.Future] = None
    model: Optional[aio.Future] = None
    mrq_model: Optional[aio.Future] = None

    def __del__(self):
        for future in [self.bandit, self.model]:
            if isinstance(future, aio.Future):
                future.cancel()


async def allocated_subgraph_watcher():
    update_loops: Dict[str, SubgraphUpdateLoops] = dict()
    excluded_subgraphs = set(
        (args.relative_query_costs_exclude_subgraphs or "").split(",")
    )

    # Initialize connection pool to PG database
    try:
        conn_string = (
            f"host={args.postgres_host} "
            f"dbname={args.postgres_database} "
            f"user={args.postgres_username} "
            f'password="{args.postgres_password}" '
            f"port={args.postgres_port}"
        )

        pgpool = psycopg_pool.AsyncConnectionPool(
            conn_string, min_size=1, max_size=args.postgres_max_connections, open=False
        )
        await pgpool.open()
        await pgpool.wait()
        assert pgpool
    except:
        logging.exception(
            "Error while creating connection pool to the PostgreSQL database."
        )
        raise

    # Initialize the extra table
    logsDB = LogsDB(pgpool)
    await logsDB.create_mrq_log_table()

    # Initialize indexer-service metrics endpoints
    if args.indexer_service_metrics_endpoint:  # static list
        metrics_endpoints = StaticMetricsEndpoints(
            args.indexer_service_metrics_endpoint
        )
    else:  # auto from k8s
        metrics_endpoints = K8SServiceWatcherMetricsEndpoints(
            args.indexer_service_metrics_k8s_service
        )

    while True:
        try:
            allocated_subgraphs = (await get_allocated_subgraphs()) - excluded_subgraphs
        except:
            logging.exception(
                "Exception occurred while getting the currently allocated subgraphs."
            )
        else:
            # Look for new subgraphs being allocated to
            for new_subgraph in allocated_subgraphs - update_loops.keys():
                # We have to manually create the entry to be sure we're keeping track of
                # the subgraph.
                update_loops[new_subgraph] = SubgraphUpdateLoops()

                # Set the default model and variables first
                await set_cost_model(
                    new_subgraph,
                    variables=DEFAULT_AGORA_VARIABLES,
                )

                await apply_default_model(new_subgraph)

                if args.relative_query_costs:
                    # Launch the model update loop for the new subgraph
                    update_loops[new_subgraph].model = aio.ensure_future(
                        model_update_loop(new_subgraph, pgpool)
                    )
                    logging.info(
                        "Added model update loop for subgraph %s", new_subgraph
                    )
                if args.multi_root_queries:
                    # Add the multi root queries (mrq)
                    update_loops[new_subgraph].mrq_model = aio.ensure_future(
                        mrq_model_update_loop(
                            new_subgraph, pgpool
                        )  # Last parameter as true to enable mrq
                    )
                    logging.info(
                        "Added model update loop for subgraph %s", new_subgraph
                    )

                # Launch the price multiplier update loop for the new subgraph
                update_loops[new_subgraph].bandit = aio.ensure_future(
                    price_bandit_loop(new_subgraph, pgpool, metrics_endpoints)
                )
                logging.info(
                    "Added price multiplier update loop for subgraph %s", new_subgraph
                )

            # Look for subgraph not being allocated to anymore
            for removed_subgraph in update_loops.keys() - allocated_subgraphs:
                del update_loops[removed_subgraph]

        await aio.sleep(30)


async def metrics_server():
    global terminate

    metrics_server = await start_http_server(port=8000)
    while True:
        await aio.sleep(1)
    await metrics_server.close()


def main():
    init_config()
    future = aio.gather(allocated_subgraph_watcher(), metrics_server())
    aio.get_event_loop().run_until_complete(future)


if __name__ == "__main__":
    main()
