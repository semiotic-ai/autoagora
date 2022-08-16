# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import inspect
import argparse
from typing import Union

from agents.reinforcement_learning_bandit import (
    ProximalPolicyOptimizationBandit,
    RollingMemContinuousBandit,
    VanillaPolicyGradientBandit,
)
from agents.heuristic_agents import RandomAgent
from agents.action_mixins import ActionMixin, ScaledActionMixin

_AGENT_TYPES = {
    "VanillaPolicyGradientBandit": VanillaPolicyGradientBandit,
    "vpg": VanillaPolicyGradientBandit,
    "ProximalPolicyOptimizationBandit": ProximalPolicyOptimizationBandit,
    "ppo": ProximalPolicyOptimizationBandit,
    "RollingMemContinuousBandit": RollingMemContinuousBandit,
    "rolling_ppo": RollingMemContinuousBandit,
    "RandomAgent": RandomAgent,
    "random": RandomAgent,
}


class AgentFactory(object):
    """Factory creating agents.

    Args:
        agent_type: Type of the agent (Options: "vpg", "ppo", "rolling_ppo")
        args: List of arguments passed to agent constructor.
        kwargs: Dict of keyword arguments passed to agent constructor.
    """

    def __new__(
        cls, agent_type: str, *args, **kwargs
    ) -> Union[
        ProximalPolicyOptimizationBandit,
        RollingMemContinuousBandit,
        VanillaPolicyGradientBandit,
    ]:
        # Get base type.
        base_agent_class = _AGENT_TYPES[agent_type]
        # Check action scaling (use scaling by definition => True).
        use_scaling = kwargs.pop("use_scaling", True)
        if use_scaling:
            mixin_class = ScaledActionMixin
        else:
            mixin_class = ActionMixin

        # Create init method for the extended class.
        def extended_init(self, **kwargs):
            # Process extended kwargs - skip "self."
            exts_init_args = inspect.getfullargspec(mixin_class.__init__).args[1:]
            ext_kwargs = {}
            for arg in exts_init_args:
                if arg in kwargs.keys():
                    ext_kwargs[arg] = kwargs.pop(arg)

            # Process base kwargs - skip "self."
            base_init_args = inspect.getfullargspec(base_agent_class.__init__).args[1:]
            base_kwargs = {}
            for arg in base_init_args:
                if arg in kwargs.keys():
                    base_kwargs[arg] = kwargs.pop(arg)

            # Check remaining args.
            if len(kwargs) > 0:
                raise ValueError(f"Invalid arguments {kwargs} for agent x")

            # Call constructors in the right order. 
            mixin_class.__init__(self, **ext_kwargs)
            base_agent_class.__init__(self, **base_kwargs)

        # Assemble the class.
        ExtendedAgentClass = type(mixin_class.__name__+base_agent_class.__name__,
            (base_agent_class, mixin_class), 
            {"__init__": extended_init})

        # Create agent instance.
        agent = ExtendedAgentClass(*args, **kwargs)
        return agent


def add_agent_argparse(parser: argparse.ArgumentParser):
    """Adds argparse arguments related to agent to parser."""
    parser.add_argument(
        "-b",
        "--buffer-size",
        default=10,
        type=int,
        help="Sets agent's buffer size (DEFAULT: 10)",
    )
    parser.add_argument(
        "-l",
        "--learning-rate",
        default=0.01,
        type=float,
        help="Sets the learning rate (DEFAULT: 0.01)",
    )
