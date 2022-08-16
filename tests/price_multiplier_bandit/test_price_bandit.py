# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from statistics import mean
from typing import Type, Union

import numpy
import pytest
import torch

from agents.continuous_action_agent import ContinuousActionBandit
from agents.reinforcement_learning_bandit import (
    ProximalPolicyOptimizationBandit,
    RollingMemContinuousBandit,
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
            learning_rate=0.1, initial_mean=gauss_mean, initial_logstddev=1.0
        )
        # Get number of bids and average.
        bids = 0.0
        for _ in range(1000):
            bids += bandit.get_bids()
        mean_bids = bids / 1000
        assert (mean_bids >= gauss_min) and (mean_bids <= gauss_max)
