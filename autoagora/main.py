# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import asyncio as aio
import logging
from dataclasses import dataclass
from typing import Dict, Optional

import configargparse
from prometheus_async.aio.web import start_http_server

from autoagora.config import init_config
from autoagora.indexer_utils import get_allocated_subgraphs, set_cost_model
from autoagora.price_multiplier import price_bandit_loop

# Import the model builder only when "--experimental-model-builder"
argsparser = configargparse.get_arg_parser()
argsparser.add_argument(
    "--experimental-model-builder",
    env_var="EXPERIMENTAL_MODEL_BUILDER",
    action="store_true",
    help="Activates the relative query cost discovery. Otherwise only builds a default "
    "query pricing model with automated market price discovery.",
)
parsed_args, remaining_args = argsparser.parse_known_args()
experimental_model_builder = parsed_args.experimental_model_builder
del parsed_args
if experimental_model_builder:
    from autoagora.model_builder import model_update_loop

DEFAULT_AGORA_VARIABLES = {"DEFAULT_COST": 50}


init_config(remaining_args)


@dataclass
class SubgraphUpdateLoops:
    bandit: Optional[aio.Future] = None
    model: Optional[aio.Future] = None

    def __del__(self):
        for future in [self.bandit, self.model]:
            if isinstance(future, aio.Future):
                future.cancel()


async def allocated_subgraph_watcher():
    update_loops: Dict[str, SubgraphUpdateLoops] = dict()

    while True:
        try:
            allocated_subgraphs = await get_allocated_subgraphs()
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
                    model="default => $DEFAULT_COST * $GLOBAL_COST_MULTIPLIER;",
                    variables=DEFAULT_AGORA_VARIABLES,
                )

                if experimental_model_builder:
                    # Launch the model update loop for the new subgraph
                    update_loops[new_subgraph].model = aio.ensure_future(
                        model_update_loop(new_subgraph)
                    )
                    logging.info(
                        "Added model update loop for subgraph %s", new_subgraph
                    )

                # Launch the price multiplier update loop for the new subgraph
                update_loops[new_subgraph].bandit = aio.ensure_future(
                    price_bandit_loop(new_subgraph)
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
    future = aio.gather(allocated_subgraph_watcher(), metrics_server())
    aio.get_event_loop().run_until_complete(future)


if __name__ == "__main__":
    main()
