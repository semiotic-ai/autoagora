# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

from price_multiplier_bandit.environment_factory import EnvironmentFactory
from price_multiplier_bandit.simulated_subgraph import (
    Environment,
    NoisyCyclicQueriesSubgraph,
    NoisyQueriesSubgraph,
)


class TestEnvironmentFactory:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "env_type,env_class",
        [
            ("NoisyQueriesSubgraph", NoisyQueriesSubgraph),
            ("noisy_static", NoisyQueriesSubgraph),
            ("NoisyCyclicQueriesSubgraph", NoisyCyclicQueriesSubgraph),
            ("noisy_cyclic", NoisyCyclicQueriesSubgraph),
        ],
    )
    def test_env_proper_types(self, env_type: str, env_class: Environment):
        """Test whether the factory returns proper types."""
        env = EnvironmentFactory(env_type)
        assert type(env) == env_class

    @pytest.mark.unit
    def test_env_invalid_types(self):
        """Test whether the factory raises error with improper type."""
        with pytest.raises(KeyError):
            _ = EnvironmentFactory("invalid")

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "env_type,env_class,noise",
        [
            ("noisy_static", NoisyQueriesSubgraph, True),
            ("static", NoisyQueriesSubgraph, False),
            ("noisy_cyclic", NoisyCyclicQueriesSubgraph, True),
            ("cyclic", NoisyQueriesSubgraph, False),
        ],
    )
    def test_env_noise(self, env_type: str, env_class: Environment, noise: bool):
        """Test whether the factory properly handles environment noise."""
        env = EnvironmentFactory(env_type)
        assert env._noise == noise
