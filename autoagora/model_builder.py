# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import asyncio as aio
import logging

import configargparse
import graphql

from autoagora.config import args
from autoagora.indexer_utils import set_cost_model
from autoagora.logs_db import LogsDB

argsparser = configargparse.get_argument_parser()
argsparser.add_argument(
    "--agora-models-refresh-interval",
    env_var="AGORA_MODELS_REFRESH_INTERVAL",
    required=False,
    type=int,
    default=3600,
    help="Interval in seconds between rebuilds of the Agora models.",
)


async def model_builder(subgraph: str) -> str:
    logs_db = LogsDB()
    await logs_db.connect()

    agora_entries = []

    most_frequent_queries = await logs_db.get_most_frequent_queries(subgraph)

    for frequent_query in most_frequent_queries:
        # Keep only query body -- ie. no var defs
        query = graphql.parse(frequent_query.query)
        assert len(query.definitions) == 1  # Should be single root query
        query = query.definitions[0].selection_set  # type: ignore
        query = "query " + graphql.print_ast(query)

        agora_entries += [
            "\n".join(
                (
                    f"# count:        {frequent_query.count}",
                    f"# min time:     {frequent_query.min_time}",
                    f"# max time:     {frequent_query.max_time}",
                    f"# avg time:     {frequent_query.avg_time}",
                    f"# stddev time:  {frequent_query.stddev_time}",
                    f"{query} => {frequent_query.avg_time}"
                    f" * $GLOBAL_COST_MULTIPLIER;",
                )
            )
        ]

    agora_entries += [
        f"default => $DEFAULT_COST * $GLOBAL_COST_MULTIPLIER;",
    ]

    model = "\n\n".join(agora_entries)
    logging.debug("Generated Agora model: \n%s", model)

    return model


async def model_update_loop(subgraph: str):
    while True:
        model = await model_builder(subgraph)
        await set_cost_model(subgraph, model)
        await aio.sleep(args.agora_models_refresh_interval)
