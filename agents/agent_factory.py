# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
from typing import Union

from agents.reinforcement_learning_bandit import (
    ProximalPolicyOptimizationBandit,
    RollingMemContinuousBandit,
    VanillaPolicyGradientBandit,
)
from agents.heuristic_agents import RandomAgent, RandomScaledAgent

_AGENT_TYPES = {
    "VanillaPolicyGradientBandit": VanillaPolicyGradientBandit,
    "vpg": VanillaPolicyGradientBandit,
    "ProximalPolicyOptimizationBandit": ProximalPolicyOptimizationBandit,
    "ppo": ProximalPolicyOptimizationBandit,
    "RollingMemContinuousBandit": RollingMemContinuousBandit,
    "rolling_ppo": RollingMemContinuousBandit,
    "RandomScaledAgent": RandomScaledAgent,
    "random_scaled": RandomScaledAgent,
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
        return _AGENT_TYPES[agent_type](*args, **kwargs)


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
