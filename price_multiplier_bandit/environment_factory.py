# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from price_multiplier_bandit.simulated_subgraph import (
    NoisyCyclicQueriesSubgraph,
    NoisyQueriesSubgraph,
)

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
