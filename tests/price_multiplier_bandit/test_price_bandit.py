# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence, Type, Union

import numpy
import pytest
import torch

from price_multiplier_bandit.price_bandit import (
    ContinuousActionBandit,
    ProximalPolicyOptimizationBandit,
    RollingMemContinuousBandit,
    SafeRollingMemContinuousBandit,
    VanillaPolicyGradientBandit,
)


class TestContinuousActionBandit:
    @pytest.mark.unit
    @pytest.mark.parametrize("test_input", [0.3, torch.tensor(0.3)])
    def test_scale(self, test_input: Union[float, torch.Tensor]):
        """Test the scale - inv_scale operation."""
        assert numpy.isclose(
            test_input,
            ContinuousActionBandit.inv_scale(ContinuousActionBandit.scale(test_input)),
        )

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "bandit_class",
        [
            VanillaPolicyGradientBandit,
            RollingMemContinuousBandit,
            ProximalPolicyOptimizationBandit,
        ],
    )
    @pytest.mark.parametrize(
        "gauss_mean, gauss_min, gauss_max",
        [
            [0.0, -1.0, 1.0],
            [10.0, 9.5, 10.5],
        ],
    )
    def test_zero_mean_bead(
        self,
        bandit_class: Type[ContinuousActionBandit],
        gauss_mean: float,
        gauss_min: float,
        gauss_max: float,
    ):
        """Tests bid"""
        # Create agent.
        bandit = bandit_class(
            learning_rate=0.1, initial_mean=gauss_mean, initial_logstddev=0.0
        )
        # Get number of bids and average.
        bids = 0.0
        for _ in range(1000):
            bids += bandit.get_bids()
        mean_bids = bids / 1000
        assert (mean_bids >= gauss_min) and (mean_bids <= gauss_max)

    @pytest.mark.unit
    def test_reward_buffer_zero_true(self):
        bandit = SafeRollingMemContinuousBandit(learning_rate=1e-3, fallback_price_multiplier=1e-6)

        rewards = [0.0 for _ in range(50)]
        losses = []

        for i, reward in enumerate(rewards):
            bandit.get_action()
            bandit.add_reward(reward)
            losses += [bandit.update_policy()]

        assert all(map(lambda v: v is None, losses))

    @pytest.mark.unit
    def test_reward_buffer_zero_false(self):
        bandit = SafeRollingMemContinuousBandit(learning_rate=1e-3, fallback_price_multiplier=1e-6)

        rewards = numpy.random.random(50)
        losses = []

        for i, reward in enumerate(rewards):
            bandit.get_action()
            bandit.add_reward(reward)
            losses += [bandit.update_policy()]

        assert not all(map(lambda v: v is None, losses))

    @pytest.mark.unit
    def test_reward_buffer_never_zero_false(self):
        bandit = SafeRollingMemContinuousBandit(learning_rate=1e-3, fallback_price_multiplier=1e-6)

        rewards = numpy.random.random(50) + 0.1
        rewards[42] = 0.0

        for reward in rewards:
            bandit.get_action()
            bandit.add_reward(reward)
            bandit.update_policy()

        assert not bandit.is_reward_buffer_never_zero()

    @pytest.mark.unit
    def test_reward_buffer_never_zero_true(self):
        bandit = SafeRollingMemContinuousBandit(learning_rate=1e-3, fallback_price_multiplier=1e-6)

        rewards = numpy.random.random(50) + 0.1

        for reward in rewards:
            bandit.get_action()
            bandit.add_reward(reward)
            bandit.update_policy()

        assert bandit.is_reward_buffer_never_zero()
