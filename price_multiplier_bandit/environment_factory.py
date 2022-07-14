# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from price_multiplier_bandit.simulated_subgraph import (
    NoisyCyclicQueriesSubgraph,
    NoisyQueriesSubgraph,
)
import argparse

_ENVIRONMENT_TYPES = {
    "NoisyQueriesSubgraph": NoisyQueriesSubgraph,
    "noisy_static": NoisyQueriesSubgraph,
    "NoisyCyclicQueriesSubgraph": NoisyCyclicQueriesSubgraph,
    "noisy_cyclic": NoisyCyclicQueriesSubgraph,
}

class EnvironmentFactory(object):
    """Factory creating environments.
    
    Args:
        environment_type: Type of the environment (Options: "noisy_cyclic", "noisy_static")
        args: List of arguments passed to agent constructor.
        kwargs: Dict of keyword arguments passed to agent constructor.
    """
    def __new__( cls, environment_type: str, *args, **kwargs):
        return _ENVIRONMENT_TYPES[environment_type](*args, **kwargs)


def add_environment_argparse(parser: argparse):
    """Adds argparse arguments related to environment to parser."""
    parser.add_argument(
        "-e", "--environment", default="noisy_cyclic", help="Sets the environment type (DEFAULT: noisy_cyclic)"
    )
    parser.add_argument(
        "-n", "--no-noise", action='store_false', help="Turns the noise on/off (DEFAULT: noise on)"
    )
