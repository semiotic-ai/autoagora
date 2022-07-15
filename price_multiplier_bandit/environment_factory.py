# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
from typing import Union

from price_multiplier_bandit.simulated_subgraph import (
    Environment,
    NoisyCyclicQueriesSubgraph,
    NoisyQueriesSubgraph,
)

_ENVIRONMENT_TYPES = {
    "NoisyQueriesSubgraph": NoisyQueriesSubgraph,
    "static": NoisyQueriesSubgraph,
    "noisy_static": NoisyQueriesSubgraph,
    "NoisyCyclicQueriesSubgraph": NoisyCyclicQueriesSubgraph,
    "cyclic": NoisyCyclicQueriesSubgraph,
    "noisy_cyclic": NoisyCyclicQueriesSubgraph,
}


class EnvironmentFactory(object):
    """Factory creating environments.

    Args:
        environment_type: Type of the environment (Options: "static", "noisy_static", "cyclic", "noisy_cyclic")
        args: List of arguments passed to agent constructor.
        kwargs: Dict of keyword arguments passed to agent constructor.
    """

    def __new__(
        cls, environment_type: str, *args, **kwargs
    ) -> Union[NoisyQueriesSubgraph, NoisyQueriesSubgraph]:
        # If argument is set - do nothing.
        if "noise" not in kwargs.keys():
            # If not, try to extract "noise" value from the name.
            if "noisy" in environment_type:
                kwargs["noise"] = True
            else:
                kwargs["noise"] = False
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
