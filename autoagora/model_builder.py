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
from jinja2 import Template

from autoagora.config import args
from autoagora.graph_node_utils import query_graph_node
from autoagora.indexer_utils import set_cost_model
from autoagora.logs_db import LogsDB
from autoagora.utils.constants import AGORA_ENTRY_TEMPLATE, MU, SIGMA


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
        await measure_query_time(subgraph, mrq_info, logs_db)
    # Call tables with info created
    most_frequent_queries = await logs_db.get_most_frequent_queries(
        subgraph, mrq_table=True
    )
    model = build_template(subgraph, most_frequent_queries)
    return model


# Obtains the execution time for n amount of random queries
async def measure_query_time(
    subgraph: str,
    multi_root_query_info: LogsDB.MRQ_Info,
    logs_db: LogsDB,
    iterations: int = 100,
):
    # Call db to obtain related variable lists
    query_log_ids = await logs_db.get_query_logs_id(multi_root_query_info.hash)
    for _ in range(iterations):

        query_id = random.choice(query_log_ids)[0]
        query_variables = await logs_db.get_query_variables(query_id)
        query_variables_dict = {f"_{i}": var for i, var in enumerate(query_variables)}
        start_q_execution = time.monotonic()
        await query_graph_node(multi_root_query_info.query, query_variables_dict)
        multi_root_query_info.query_time_ms = int(
            (time.monotonic() - start_q_execution) * 1000
        )
        multi_root_query_info.timestamp = datetime.now()
        await logs_db.save_generated_aa_query_values(
            multi_root_query_info, subgraph, query_variables
        )


async def mrq_model_update_loop(subgraph: str, pgpool):
    while True:
        model = await mrq_model_builder(subgraph, pgpool)
        await set_cost_model(subgraph, model)
        await aio.sleep(args.relative_query_costs_refresh_interval)


async def model_update_loop(subgraph: str, pgpool):
    while True:
        model = await model_builder(subgraph, pgpool)
        await set_cost_model(subgraph, model)
        # TODO: apply here lognormvariate , need to find a value that works
        await aio.sleep(random.lognormvariate(MU, SIGMA))


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
