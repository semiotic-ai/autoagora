# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
from typing import Any, Optional, Sequence

import configargparse
from pythonjsonlogger import jsonlogger


class _Args(configargparse.Namespace):
    def __getattribute__(self, name) -> Any:
        return object.__getattribute__(self, name)


# `args` will contain the global argument values after calling `init_config()`
args = _Args()


def init_config(argv: Optional[Sequence[str]] = None):
    """Parses the arguments from the global argument parser and sets logging config.

    Add new arguments in this function.

    Argument values are added to the global `args` namespace object declared in this
    module.
    """
    # 2 arg parsers to make it possible to fetch the `experimental_model_builder` option
    # early without breaking anything.
    argsparser_experimental_model_builder = configargparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False
    )
    argsparser_experimental_model_builder.add_argument(
        "--experimental-model-builder",
        env_var="EXPERIMENTAL_MODEL_BUILDER",
        action="store_true",
        help="Activates the relative query cost discovery. Otherwise only builds a "
        "default query pricing model with automated market price discovery.",
    )
    # Get the value of `experimental-model-builder` early
    _, argv = argsparser_experimental_model_builder.parse_known_args(
        argv, namespace=args
    )

    # 2nd arg parser
    argsparser = configargparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[argsparser_experimental_model_builder],
    )

    #
    # General arguments
    #
    argsparser.add_argument(
        "--log-level",
        env_var="LOG_LEVEL",
        type=str,
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="WARNING",
        required=False,
    )
    argsparser.add_argument(
        "--json-logs",
        env_var="JSON_LOGS",
        type=bool,
        default=False,
        required=False,
        help="Output logs in JSON format. Compatible with GKE.",
    )

    #
    # Indexer utils
    #
    argsparser.add_argument(
        "--indexer-agent-mgmt-endpoint",
        env_var="INDEXER_AGENT_MGMT_ENDPOINT",
        required=True,
        help="URL to the indexer-agent management GraphQL endpoint.",
    )

    #
    # Query volume metrics
    #

    argsparser.add_argument(
        "--indexer-service-metrics-endpoint",
        env_var="INDEXER_SERVICE_METRICS_ENDPOINT",
        required=True,
        help="HTTP endpoint for the indexer-service metrics.",
    )

    #
    # Price multiplier (Absolute price)
    #
    argsparser.add_argument(
        "--observation-duration",
        env_var="MEASUREMENT_PERIOD",
        required=False,
        type=int,
        default=60,
        help="Duration of the measurement period of the query-per-second after a price "
        "multiplier update.",
    )

    #
    # Optional model builder (Relative query costs)
    #
    model_builder_group = argsparser.add_argument_group(
        title="Model Builder Options", description=""
    )
    model_builder_group.add_argument(
        "--exclude-subgraphs",
        env_var="EXCLUDE_SUBGRAPHS",
        required=False,
        help="Comma delimited list of subgraphs (ipfs hash) to exclude from model "
        "updates.",
    )
    model_builder_group.add_argument(
        "--agora-models-refresh-interval",
        env_var="AGORA_MODELS_REFRESH_INTERVAL",
        required=False,
        type=int,
        default=3600,
        help="Interval in seconds between rebuilds of the Agora models.",
    )

    #
    # Logs DB
    #

    # Needed only if the model builder is turned on
    model_builder_group.add_argument(
        "--logs-postgres-host",
        env_var="LOGS_POSTGRES_HOST",
        required=args.experimental_model_builder,
        help="Host of the postgres instance storing the logs.",
    )
    model_builder_group.add_argument(
        "--logs-postgres-port",
        env_var="LOGS_POSTGRES_PORT",
        required=False,
        type=int,
        default=5432,
        help="Port of the postgres instance storing the logs.",
    )
    model_builder_group.add_argument(
        "--logs-postgres-database",
        env_var="LOGS_POSTGRES_DATABASE",
        required=args.experimental_model_builder,
        help="Name of the logs database.",
    )
    model_builder_group.add_argument(
        "--logs-postgres-username",
        env_var="LOGS_POSTGRES_USERNAME",
        required=args.experimental_model_builder,
        help="Username for the logs database.",
    )
    model_builder_group.add_argument(
        "--logs-postgres-password",
        env_var="LOGS_POSTGRES_PASSWORD",
        required=args.experimental_model_builder,
        help="Password for the logs database.",
    )

    argsparser.parse_args(args=argv, namespace=args)

    # Set the logs formatting
    if args.json_logs:
        logHandler = logging.StreamHandler()
        formatter = jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
            rename_fields={
                "asctime": "date",
                "name": "subcomponent",
                "levelname": "level",
            },
        )
        logHandler.setFormatter(formatter)
        logging.root.addHandler(logHandler)
    else:
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    logging.root.setLevel(args.log_level)
