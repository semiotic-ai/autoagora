# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import asyncio as aio
import logging
import os
from importlib.metadata import version

import asyncpg
from jinja2 import Template
from platformdirs import user_config_dir

from autoagora.config import args
from autoagora.indexer_utils import set_cost_model
from autoagora.logs_db import LogsDB
from autoagora.utils.constants import AGORA_ENTRY_TEMPLATE


async def model_builder(subgraph: str, pgpool: asyncpg.Pool) -> str:
    logs_db = LogsDB(pgpool)
    aa_version = version("autoagora")
    most_frequent_queries = await logs_db.get_most_frequent_queries(subgraph)
    manual_agora_entry = obtain_manual_entries()
    template = Template(AGORA_ENTRY_TEMPLATE)
    model = template.render(
        aa_version=aa_version,
        most_frequent_queries=most_frequent_queries,
        manual_entry=manual_agora_entry,
    )
    logging.debug("Generated Agora model: \n%s", model)
    return model


async def model_update_loop(subgraph: str, pgpool):
    while True:
        model = await model_builder(subgraph, pgpool)
        await set_cost_model(subgraph, model)
        await aio.sleep(args.relative_query_costs_refresh_interval)


def obtain_manual_entries():
    # Obtain path of the python file
    agora_models_dir = user_config_dir(args.aa_app_manual_entry_path)
    full_manual_entries = ""
    if os.path.isdir(agora_models_dir):
        manual_agora_entries = os.listdir(agora_models_dir)
        for agora_entry in manual_agora_entries:
            if agora_entry.endswith(".agora"):
                agora_entry_full_path = agora_models_dir + "/" + agora_entry
                with open(agora_entry_full_path, "r") as file:
                    manual_agora_model = file.read()
                    full_manual_entries += manual_agora_model
        if full_manual_entries != "":
            return full_manual_entries
        else:
            return None
    return None
