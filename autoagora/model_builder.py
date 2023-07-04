# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import asyncio as aio
import logging
import os
from importlib.metadata import version

import asyncpg
from jinja2 import Template

from autoagora.config import args
from autoagora.indexer_utils import set_cost_model
from autoagora.logs_db import LogsDB
from autoagora.utils.constants import AGORA_ENTRY_TEMPLATE


async def model_builder(subgraph: str, pgpool: asyncpg.Pool) -> str:
    logs_db = LogsDB(pgpool)
    most_frequent_queries = await logs_db.get_most_frequent_queries(subgraph)
    model = build_template(subgraph, most_frequent_queries)
    return model


async def manual_model_builder(subgraph: str):
    model = build_template(subgraph)
    await set_cost_model(subgraph, model)


async def model_update_loop(subgraph: str, pgpool):
    while True:
        model = await model_builder(subgraph, pgpool)
        await set_cost_model(subgraph, model)
        await aio.sleep(args.relative_query_costs_refresh_interval)


def build_template(subgraph: str, most_frequent_queries = []):
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
        logging.debug(
            "No path for manual agora entries was given for subgraph %s", subgraph
        )
        return None

    agora_entry_full_path = os.path.join(agora_models_dir, subgraph + ".agora")
    if os.path.isfile(agora_entry_full_path):
        with open(agora_entry_full_path, "r") as file:
            manual_agora_model = file.read()
            # Just a safe measure for empty files
            if manual_agora_model != "":
                logging.debug("Manual model was loaded for subgraph %s", subgraph)
                return manual_agora_model
    return None
