# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

from agents.action_mixins import (
    Action,
    DeterministicAction,
    GaussianAction,
    ScaledGaussianAction,
)
from agents.agent_factory import AgentFactory
from agents.policy_mixins import NoUpdatePolicy, Policy
from agents.reinforcement_learning_policy_mixins import (
    ProximalPolicyOptimization,
    RollingMemoryPPO,
    VanillaPolicyGradient,
)


class TestAgentFactory:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "policy_name,policy_class",
        [
            ("vpg", VanillaPolicyGradient),
            ("ppo", ProximalPolicyOptimization),
            ("rolling_ppo", RollingMemoryPPO),
            ("no_update", NoUpdatePolicy),
        ],
    )
    def test_agent_policy_proper_types(self, policy_name: str, policy_class: Policy):
        """Test whether the factory returns agents with proper policies."""
        agent = AgentFactory(
            agent_name="test_agent", agent_section={"policy": policy_name}
        )
        assert policy_class in agent.__class__.__bases__

    @pytest.mark.unit
    def test_agent_policy_invalid_types(self):
        """Test whether the factory raises error with improper policy type."""
        with pytest.raises(KeyError):
            _ = AgentFactory(
                agent_name="invalid_agent", agent_section={"policy": "invalid"}
            )

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "action_name,action_class",
        [
            ("gaussian", GaussianAction),
            ("scaled_gaussian", ScaledGaussianAction),
            ("deterministic", DeterministicAction),
        ],
    )
    def test_agent_action_proper_types(self, action_name: str, action_class: Action):
        """Test whether the factory returns agents with proper policies."""
        agent = AgentFactory(
            agent_name="test_agent", agent_section={"action": action_name}
        )
        assert action_class in agent.__class__.__bases__

    @pytest.mark.unit
    def test_agent_action_invalid_types(self):
        """Test whether the factory raises error with improper policy type."""
        with pytest.raises(KeyError):
            _ = AgentFactory(
                agent_name="invalid_agent", agent_section={"action": "invalid"}
            )
