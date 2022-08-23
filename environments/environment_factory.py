# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse

from environments.environment import Environment
from environments.shared_subgraph import NoisySharedSubgraph
from environments.simulated_subgraph import (
    NoisyCyclicQueriesSubgraph,
    NoisyCyclicZeroQueriesSubgraph,
    NoisyDynamicQueriesSubgraph,
    NoisyQueriesSubgraph,
)

_ENVIRONMENT_TYPES = {
    "NoisyQueriesSubgraph": NoisyQueriesSubgraph,
    "static": NoisyQueriesSubgraph,
    "NoisyCyclicQueriesSubgraph": NoisyCyclicQueriesSubgraph,
    "cyclic": NoisyCyclicQueriesSubgraph,
    "NoisyCyclicZeroQueriesSubgraph": NoisyCyclicZeroQueriesSubgraph,
    "cyclic_zero": NoisyCyclicZeroQueriesSubgraph,
    "NoisyDynamicQueriesSubgraph": NoisyDynamicQueriesSubgraph,
    "dynamic": NoisyDynamicQueriesSubgraph,
    "NoisySharedSubgraph": NoisySharedSubgraph,
    "shared": NoisySharedSubgraph,
}


class EnvironmentFactory(object):
    """Factory creating environments.

    Args:
        environment_type: Type of the environment (Options: "static", "noisy_static", "cyclic", "noisy_cyclic")
        args: List of arguments passed to agent constructor.
        kwargs: Dict of keyword arguments passed to agent constructor.
    """

    def __new__(cls, environment_type: str, *args, **kwargs) -> Environment:
        # Create the environment object.
        return _ENVIRONMENT_TYPES[environment_type](*args, **kwargs)


def add_environment_argparse(parser: argparse.ArgumentParser):
    """Adds argparse arguments related to environment to parser."""
    parser.add_argument(
        "-e",
        "--environment",
        default="noisy_cyclic",
        help="Sets the environment type (DEFAULT: noisy_cyclic)",
    )
