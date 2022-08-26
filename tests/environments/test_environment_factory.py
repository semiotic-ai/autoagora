# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

from environments.environment_factory import EnvironmentFactory
from environments.simulated_subgraph import (
    Environment,
    NoisyCyclicQueriesSubgraph,
    NoisyQueriesSubgraph,
    NoisySimulatedSubgraph,
)


class TestEnvironmentFactory:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "env_type,env_class",
        [
            ("NoisyQueriesSubgraph", NoisyQueriesSubgraph),
            ("static", NoisyQueriesSubgraph),
            ("NoisyCyclicQueriesSubgraph", NoisyCyclicQueriesSubgraph),
            ("cyclic", NoisyCyclicQueriesSubgraph),
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
        "env_type,noise",
        [
            ("static", False),
            ("cyclic", False),
        ],
    )
    def test_env_noise(self, env_type: str, noise: bool):
        """Test whether the factory properly handles environment noise.

        Works only on instances of NoisyQueriesSubgraph.
        """
        env = EnvironmentFactory(env_type, noise=noise)
        # We need the environment to be SimulatedSubgraph
        assert isinstance(env, NoisySimulatedSubgraph)

        assert env._noise == noise
