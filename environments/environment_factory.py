# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
from typing import Dict, Type, Union

from environments.environment import Environment, MissingOptionalEnvironment
from environments.shared_subgraph import NoisySharedSubgraph
from environments.simulated_subgraph import (
    NoisyCyclicQueriesSubgraph,
    NoisyCyclicZeroQueriesSubgraph,
    NoisyDynamicQueriesSubgraph,
    NoisyQueriesSubgraph,
)

_ENVIRONMENT_TYPES: Dict[
    str, Union[Type[Environment], Type[MissingOptionalEnvironment]]
] = dict()

try:
    from autoagora_isa.isa import IsaSubgraph  # type: ignore
except:
    _ENVIRONMENT_TYPES["isa"] = MissingOptionalEnvironment
else:
    _ENVIRONMENT_TYPES["isa"] = IsaSubgraph

_ENVIRONMENT_TYPES.update(
    {
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
)


class EnvironmentFactory(object):
    """Factory creating environments.

    Args:
        environment_type: Type of the environment (Options: "static", "noisy_static", "cyclic", "noisy_cyclic")
        args: List of arguments passed to agent constructor.
        kwargs: Dict of keyword arguments passed to agent constructor.
    """

    def __new__(cls, environment_type_name: str, *args, **kwargs) -> Environment:
        environment_type = _ENVIRONMENT_TYPES[environment_type_name]

        if issubclass(environment_type, Environment):
            return environment_type(*args, **kwargs)
        elif environment_type is MissingOptionalEnvironment:
            raise TypeError(
                f'Missing optional environment type "{environment_type_name}"'
            )
        else:
            raise RuntimeError(f'Unknown environment type "{environment_type}"')


def add_environment_argparse(parser: argparse.ArgumentParser):
    """Adds argparse arguments related to environment to parser."""
    parser.add_argument(
        "-e",
        "--environment",
        default="noisy_cyclic",
        help="Sets the environment type (DEFAULT: noisy_cyclic)",
    )
