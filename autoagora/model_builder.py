# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import asyncio as aio
import logging
import os
import random
import time
from datetime import datetime
from importlib.metadata import version

import psycopg_pool
from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport
from jinja2 import Template

from autoagora.config import args
from autoagora.indexer_utils import set_cost_model
from autoagora.logs_db import LogsDB
from autoagora.utils.constants import AGORA_ENTRY_TEMPLATE


async def model_builder(subgraph: str, pgpool: psycopg_pool.AsyncConnectionPool) -> str:
    logs_db = LogsDB(pgpool)
    most_frequent_queries = await logs_db.get_most_frequent_queries(subgraph)
    model = build_template(subgraph, most_frequent_queries)
    return model


async def apply_default_model(subgraph: str):
    model = build_template(subgraph)
    await set_cost_model(subgraph, model)


# (mrq) stands for multi root query
async def mrq_model_builder(subgraph: str, pgpool: psycopg_pool.AsyncConnectionPool) -> str:
    logs_db = LogsDB(pgpool)
    # Obtain most queried mrq
    most_frequent_multi_root_queries = (
        await logs_db.get_most_frequent_queries_null_time(subgraph)
    )
    for mrq_info in most_frequent_multi_root_queries:
        await obtain_query_time(subgraph, mrq_info, logs_db)
    # Call tables with info created
    most_frequent_queries = await logs_db.get_most_frequent_queries(
        subgraph, mrq_table=True
    )
    model = build_template(subgraph, most_frequent_queries)
    return model


# Obtains the execution time for n amount of random queries
async def obtain_query_time(
    subgraph, multi_root_query_info, logs_db, iterations: int = 100
):
    # Call db to obtain related variable lists
    query_variables_list = await logs_db.get_query_variables_from_query_hash(
        multi_root_query_info.hash
    )
    no_queries = len(query_variables_list)
    for _ in range(iterations):

        query_variables = query_variables_list[random.randint(0, no_queries)]
        query_vars = {f"_{i}": var for i, var in enumerate(query_variables)}

        async with Client(
            transport=AIOHTTPTransport(args.graph_node_query_endpoint),
            fetch_schema_from_transport=False,
        ) as session:

            start_q_execution = time.time()
            await session.execute(gql(multi_root_query_info.query), variable_values=query_vars)  # type: ignore
            multi_root_query_info.query_time_ms = int(
                (time.time() - start_q_execution) * 1000
            )

        multi_root_query_info.timestamp = datetime.now()
        logs_db.save_generated_aa_query_values(
            multi_root_query_info, subgraph, query_variables
        )


async def mrq_model_update_loop(subgraph: str, pgpool):
    while True:
        model = await mrq_model_builder(subgraph, pgpool)
        await set_cost_model(subgraph, model)
        # Should this also be this amount of time?
        await aio.sleep(args.relative_query_costs_refresh_interval)


async def model_update_loop(subgraph: str, pgpool):
    while True:
        model = await model_builder(subgraph, pgpool)
        await set_cost_model(subgraph, model)
        await aio.sleep(args.relative_query_costs_refresh_interval)


def build_template(subgraph: str, most_frequent_queries=None):
    if most_frequent_queries is None:
        most_frequent_queries = []
    aa_version = version("autoagora")
    manual_agora_entry = obtain_manual_entries(subgraph)
    template = Template(AGORA_ENTRY_TEMPLATE)
    model = template.render(
        aa_version=aa_version,
        most_frequent_queries=most_frequent_queries,
        manual_entry=manual_agora_entry,
    )
    logging.debug("Generated Agora model: \n%s", model)
    return model


def obtain_manual_entries(subgraph: str):
    # Obtain path of the python file
    agora_models_dir = args.manual_entry_path
    if agora_models_dir is None:
        return None

    agora_entry_full_path = os.path.join(agora_models_dir, subgraph + ".agora")
    if os.path.isfile(agora_entry_full_path):
        with open(agora_entry_full_path, "r") as file:
            manual_agora_model = file.read()
            # Just a safe measure for empty files
            if manual_agora_model != "":
                logging.debug("Manual model was loaded for subgraph %s", subgraph)
                return manual_agora_model
    logging.debug(
        "No path for manual agora entries was given for subgraph %s", subgraph
    )
    return None
