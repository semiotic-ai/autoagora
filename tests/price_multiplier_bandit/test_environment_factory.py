# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

from price_multiplier_bandit.simulated_subgraph import (
    Environment,
    NoisyCyclicQueriesSubgraph,
    NoisyQueriesSubgraph,
)
from price_multiplier_bandit.environment_factory import EnvironmentFactory


class TestEnvironmentFactory:
    @pytest.mark.unit
    @pytest.mark.parametrize( "env_type,env_class", [
            ("NoisyQueriesSubgraph", NoisyQueriesSubgraph),
            ("noisy_static", NoisyQueriesSubgraph),
            ("NoisyCyclicQueriesSubgraph", NoisyCyclicQueriesSubgraph),
            ("noisy_cyclic", NoisyCyclicQueriesSubgraph),
        ])
    def test_env_proper_types(self, env_type: str, env_class: Environment):
        """Test whether the factory returns proper types."""
        env = EnvironmentFactory(env_type)
        assert type(env) == env_class


    @pytest.mark.unit
    def test_env_proper_types(self):
        """Test whether the factory raises error with improper type."""
        with pytest.raises(KeyError):
            _ = EnvironmentFactory("invalid")
