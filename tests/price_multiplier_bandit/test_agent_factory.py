# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

from price_multiplier_bandit.agent_factory import AgentFactory
from price_multiplier_bandit.price_bandit import (
    ContinuousActionBandit,
    ProximalPolicyOptimizationBandit,
    RollingMemContinuousBandit,
    VanillaPolicyGradientBandit,
)


class TestAgentFactory:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "agent_type,agent_class",
        [
            ("VanillaPolicyGradientBandit", VanillaPolicyGradientBandit),
            ("vpg", VanillaPolicyGradientBandit),
            ("ProximalPolicyOptimizationBandit", ProximalPolicyOptimizationBandit),
            ("ppo", ProximalPolicyOptimizationBandit),
            ("RollingMemContinuousBandit", RollingMemContinuousBandit),
            ("rolling_ppo", RollingMemContinuousBandit),
        ],
    )
    def test_agent_proper_types(
        self, agent_type: str, agent_class: ContinuousActionBandit
    ):
        """Test whether the factory returns proper types."""
        agent = AgentFactory(agent_type, learning_rate=0.01, buffer_max_size=10)
        assert type(agent) == agent_class

    @pytest.mark.unit
    def test_agent_proper_types(self):
        """Test whether the factory raises error with improper type."""
        with pytest.raises(KeyError):
            _ = AgentFactory("invalid")
