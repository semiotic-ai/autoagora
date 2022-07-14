# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from price_multiplier_bandit.price_bandit import (
    ProximalPolicyOptimizationBandit,
    RollingMemContinuousBandit,
    VanillaPolicyGradientBandit,
)

_AGENT_TYPES = {
    "VanillaPolicyGradientBandit": VanillaPolicyGradientBandit,
    "vpg": VanillaPolicyGradientBandit,
    "ProximalPolicyOptimizationBandit": ProximalPolicyOptimizationBandit,
    "ppo": ProximalPolicyOptimizationBandit,
    "RollingMemContinuousBandit": RollingMemContinuousBandit,
    "rolling_ppo": RollingMemContinuousBandit,
}

class AgentFactory(object):
    """Factory creating agents.
    
    Args:
        agent_type: Type of the agent (Options: "vpg", "ppo", "rolling_ppo")
        args: List of arguments passed to agent constructor.
        kwargs: Dict of keyword arguments passed to agent constructor.
    """
    def __new__( cls, agent_type: str, *args, **kwargs):
        return _AGENT_TYPES[agent_type](*args, **kwargs)

if __name__ == "__main__":
    agent = AgentFactory("vpg", learning_rate=0.01, buffer_max_size=10 )