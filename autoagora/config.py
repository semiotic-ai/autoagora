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

    argparser = configargparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    #
    # General arguments
    #
    argparser.add_argument(
        "--log-level",
        env_var="LOG_LEVEL",
        type=str,
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="WARNING",
        required=False,
    )
    argparser.add_argument(
        "--json-logs",
        env_var="JSON_LOGS",
        type=bool,
        default=False,
        required=False,
        help="Output logs in JSON format. Compatible with GKE.",
    )

    #
    # AutoAgora DB
    #
    argparser_database_group = argparser.add_argument_group(
        "Database settings",
        description="Must be the same database as AutoAgora Processor's if the "
        "relative costs models generator is enabled.",
    )
    argparser_database_group.add_argument(
        "--postgres-host",
        env_var="POSTGRES_HOST",
        required=True,
        help="Host of the postgres instance to be used by AutoAgora.",
    )
    argparser_database_group.add_argument(
        "--postgres-port",
        env_var="POSTGRES_PORT",
        required=False,
        type=int,
        default=5432,
        help="Port of the postgres instance to be used by AutoAgora.",
    )
    argparser_database_group.add_argument(
        "--postgres-database",
        env_var="POSTGRES_DATABASE",
        required=False,
        default="autoagora",
        help="Name of the database to be used by AutoAgora.",
    )
    argparser_database_group.add_argument(
        "--postgres-username",
        env_var="POSTGRES_USERNAME",
        required=True,
        help="Username for the database to be used by AutoAgora.",
    )
    argparser_database_group.add_argument(
        "--postgres-password",
        env_var="POSTGRES_PASSWORD",
        required=True,
        help="Password for the database to be used by AutoAgora.",
    )
    argparser_database_group.add_argument(
        "--postgres-max-connections",
        default=1,
        type=int,
        env_var="POSTGRES_MAX_CONNECTIONS",
        required=False,
        help="Maximum postgres connections (internal pool).",
    )

    #
    # Indexer utils
    #
    argparser.add_argument(
        "--indexer-agent-mgmt-endpoint",
        env_var="INDEXER_AGENT_MGMT_ENDPOINT",
        required=True,
        help="URL to the indexer-agent management GraphQL endpoint.",
    )

    #
    # Query volume metrics
    #
    argparser.add_argument(
        "--indexer-service-metrics-endpoint",
        env_var="INDEXER_SERVICE_METRICS_ENDPOINT",
        required=True,
        help="HTTP endpoint for the indexer-service metrics. Can be a comma-separated for multiple endpoints.",
    )

    #
    # Price multiplier (Absolute price)
    #
    argparser.add_argument(
        "--qps-observation-duration",
        env_var="QPS_OBSERVATION_DURATION",
        required=False,
        type=int,
        default=60,
        help="Duration of the measurement period of the query-per-second after a price "
        "multiplier update.",
    )

    #
    # Optional model builder (Relative query costs)
    #
    argparser_relative_query_costs = argparser.add_argument_group(
        "Relative query costs generator settings"
    )
    argparser_relative_query_costs.add_argument(
        "--relative-query-costs",
        env_var="RELATIVE_QUERY_COSTS",
        action="store_true",
        help="(EXPERIMENTAL) Enables the relative query cost generator. Otherwise only "
        "builds a default query pricing model with automated market price discovery.",
    )
    argparser_relative_query_costs.add_argument(
        "--relative-query-costs-exclude-subgraphs",
        env_var="RELATIVE_QUERY_COSTS_EXCLUDE_SUBGRAPHS",
        required=False,
        help="Comma delimited list of subgraphs (ipfs hash) to exclude from the "
        "relative query costs model generator.",
    )
    argparser_relative_query_costs.add_argument(
        "--relative-query-costs-refresh-interval",
        env_var="RELATIVE_QUERY_COSTS_REFRESH_INTERVAL",
        required=False,
        type=int,
        default=3600,
        help="(Seconds) Interval between rebuilds of the relative query costs models.",
    )

    argparser.parse_args(args=argv, namespace=args)

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
