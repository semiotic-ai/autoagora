# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import inspect

import torch.optim as optim

from agents.action_mixins import (
    DeterministicAction,
    GaussianAction,
    ScaledGaussianAction,
)
from agents.agent import Agent
from agents.policy_mixins import NoUpdatePolicy
from agents.reinforcement_learning_policy_mixins import (
    ProximalPolicyOptimization,
    RollingMemoryPPO,
    VanillaPolicyGradient,
)

_POLICY_TYPES = {
    "VanillaPolicyGradient": VanillaPolicyGradient,
    "vpg": VanillaPolicyGradient,
    "ProximalPolicyOptimization": ProximalPolicyOptimization,
    "ppo": ProximalPolicyOptimization,
    "RollingMemoryPPO": RollingMemoryPPO,
    "rolling_ppo": RollingMemoryPPO,
    "NoUpdatePolicy": NoUpdatePolicy,
    "no_update": NoUpdatePolicy,
}

_ACTION_TYPES = {
    "ScaledGaussianAction": ScaledGaussianAction,
    "scaled_gaussian": ScaledGaussianAction,
    "GaussianAction": GaussianAction,
    "gaussian": GaussianAction,
    "DeterministicAction": DeterministicAction,
    "deterministic": DeterministicAction,
}

_OPTIMIZER_TYPES = {
    "Adam": optim.Adam,
    "adam": optim.Adam,
    "AdamW": optim.AdamW,
    "adamw": optim.AdamW,
}


class AgentFactory(object):
    """Factory creating agents by composing policy, action and optimizer.

    Args:
        agent_type: Type of the agent (Options: "vpg", "ppo", "rolling_ppo")
        args: List of arguments passed to agent constructor.
        kwargs: Dict of keyword arguments passed to agent constructor.
    """

    def __new__(cls, agent_name: str, agent_section) -> Agent:
        # Get policy section.
        policy_section = agent_section.pop("policy", {})
        # Enable "policy: type" construct.
        if type(policy_section) is str:
            policy_section = {"type": policy_section}
        # Use no updte policy by default.
        policy_class = _POLICY_TYPES[policy_section.pop("type", "NoUpdatePolicy")]

        # Get action section.
        action_section = agent_section.pop("action", {})
        # Enable "action: type" construct.
        if type(action_section) is str:
            action_section = {"type": action_section}
        # Use scaled gaussian action by default.
        action_class = _ACTION_TYPES[action_section.pop("type", "ScaledGaussianAction")]

        # Get optimizer section.
        optim_section = agent_section.pop("optimizer", {})
        # Enable "optimizer: type" construct.
        if type(optim_section) is str:
            optim_section = {"type": optim_section}
        # Use Adam by default.
        optim_class = _OPTIMIZER_TYPES[optim_section.pop("type", "Adam")]

        # Create init method for the agent class composed of action and policy.
        def composed_agent_init(self, action_section, policy_section):

            # Call constructors in the right order.
            action_class.__init__(self, **action_section)
            policy_class.__init__(self, **policy_section)
            Agent.__init__(self, name=agent_name)

        # Assemble the class.
        ComposedAgentClass = type(
            action_class.__name__ + policy_class.__name__,
            (policy_class, action_class, Agent),
            {"__init__": composed_agent_init},
        )

        # Create agent instance.
        agent = ComposedAgentClass(action_section, policy_section)

        # Initialize optimizer - if there are any params!
        if agent.params is not None:
            agent._optimizer = optim_class(params=agent.params, **optim_section)

        return agent
