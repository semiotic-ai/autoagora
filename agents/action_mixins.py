# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from math import exp, log
from typing import Union, overload

import numpy as np
import scipy.stats as stats
import torch
from torch import distributions, nn

from agents.mixin import ABCMixin


class Action(ABCMixin):
    @abstractmethod
    def get_action(self) -> float:
        """Abstract method returning agent's action."""
        pass


class ScaledGaussianAction(Action):
    """Mixin class for agents with continuous action space represented as a gausian in the scaled space.
    Moreover, the std dev operates in a separate log space.
    Mixin provides methods for moving between external (scaled bid) and internal (bid) space,sampling actions, visualization etc.

    Args:
        initial_mean: (DEFAULT: 1e-6) initial mean in the original action (i.e. scaled bid) space.
        initial_stddev: (DEFAULT: 1e-7) initial standard deviation in the original action (i.e. scaled bid) space.
    """

    def __init__(
        self,
        initial_mean: float = 1e-6,
        initial_stddev: float = 1e-7,
    ):
        # Store init params - after projecting them to the internal "log" space.
        self._initial_mean = torch.Tensor([self.inverse_bid_scale(initial_mean)])
        # TODO: think: currently this maps only to internal log space, not the desired log(log) space of stddev.
        self._initial_logstddev = torch.Tensor([self.inverse_bid_scale(initial_stddev)])

        # Store policy params - after projecting them to the internal "log" space.
        self._mean = nn.parameter.Parameter(
            torch.Tensor([self.inverse_bid_scale(initial_mean)])
        )
        # TODO: think: currently this maps only to internal log space, not the desired log(log) space of stddev.
        self._logstddev = nn.parameter.Parameter(
            torch.Tensor([self.inverse_bid_scale(initial_stddev)])
        )

    def mean(self, initial: bool = False):
        """Returns:
        Mean in the internal action (bid) space.
        """
        if initial:
            return self._initial_mean.clamp_max(self.inverse_bid_scale(1e-1))
        else:
            return self._mean.clamp_max(self.inverse_bid_scale(1e-1))

    def stddev(self, initial: bool = False):
        """Returns:
        Std dev in the internal action (bid) space.
        """
        # TODO: rething the order here -> clamp self.stddev?
        if initial:
            return self._initial_logstddev.exp()
        else:
            return self._logstddev.exp()

    @property
    def params(self):
        """Returns:
        List of trainable parameters.
        """
        return [self._mean, self._logstddev]

    def distribution(self) -> torch.distributions.Normal:
        """Returns:
        Distribution in the internal space.
        """
        return distributions.Normal(self.mean(), self.stddev())

    def get_bids(self):
        """Samples action from the action space, add it to action buffer and returns it.

        Return:
            Action sampled from the action space.
        """
        # Sample action from distribution.
        dist = self.distribution()
        action = dist.rsample().detach().item()
        # assert isinstance(action, float)

        # Add action to buffer (TODO: (re)think the decoupled buffer-scaling logic)
        if hasattr(self, "action_buffer"):
            self.action_buffer.append(action)

        return action

    def get_action(self):
        """Calls get_bids() and scale() to return scaled value."""
        bid = self.get_bids()
        scaled_bid = self.bid_scale(bid)
        return scaled_bid

    @overload
    def bid_scale(self, x: float) -> float:
        ...

    @overload
    def bid_scale(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def bid_scale(self, x: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """Scales the value."""
        if isinstance(x, float):
            try:
                # print(f"x = {x}  => exp(x) * 1e-6 = {exp(x) * 1e-6}")
                return exp(x) * 1e-6
            except OverflowError:
                # print(f"!! OverflowError in exp(x) * 1e-6 for x = {x}!!")
                exit(-1)
        elif isinstance(x, torch.Tensor):
            return x.exp() * 1e-6
        else:
            raise TypeError(f"Invalid type '{type(x)}'")

    @overload
    def inverse_bid_scale(self, x: float) -> float:
        ...

    @overload
    def inverse_bid_scale(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def inverse_bid_scale(
        self, x: Union[float, torch.Tensor]
    ) -> Union[float, torch.Tensor]:
        """Inverse operation to action (bid) scaling."""
        if isinstance(x, float):
            return log(x * 1e6)
        elif isinstance(x, torch.Tensor):
            return (x * 1e6).log()
        else:
            raise TypeError(f"Invalid type '{type(x)}'")

    async def generate_plot_data(
        self, min_x: float, max_x: float, num_points: int = 200, logspace: bool = False
    ):
        """Generates action distribution for a given cost multiplier range.

        Args:
            min_x (float): Lower bound cost multiplier.
            max_x (float): Upper bound cost multiplier.
            num_points (int, optional): Number of points. Defaults to 200.

        Returns:
            ([x1, x2, ...], [y1, y2, ...], [iy1, iy2, ...]): Triplet of lists of x, y (current policy PDF) and iy (init policy PDF).
        """

        # Prepare points in the "unscaled" x.
        if logspace:
            agent_x = np.logspace(np.log10(min_x), np.log10(max_x), num_points, base=10)
        else:
            agent_x = np.linspace(min_x, max_x, num_points)
        # Project the points into the "internal/inverted scale" x.
        agent_x_inverted_scale = [self.inverse_bid_scale(x) for x in agent_x]

        # Get agent's PDF for points in the "inverted scale".
        policy_mean = self.mean().detach().numpy()
        policy_stddev = self.stddev().detach().numpy()
        policy_y = stats.norm.pdf(agent_x_inverted_scale, policy_mean, policy_stddev)
        # Scale to 0.5.
        policy_y = 0.5 * policy_y / max(policy_y)

        # Get agent's init PDF for "unscaled" x.
        init_mean = self.mean(initial=True).detach().numpy()
        init_stddev = self.stddev(initial=True).detach().numpy()
        init_y = stats.norm.pdf(agent_x_inverted_scale, init_mean, init_stddev)
        # Scale to 0.5.
        init_y = 0.5 * init_y / max(init_y)

        # Return x, current and init policies.
        return {
            "x": agent_x,
            "policy": policy_y,
            "init policy": init_y,
        }


class GaussianAction(Action):
    """Mixin class for agents with continuous action space represented as a gausian in the regular action space (NO SCALING!)
    The std dev operates in a separate log space.
    Mixin provides methods for moving for sampling action, visualization etc.

    Args:
        initial_mean: (DEFAULT: 1e-6) initial mean in the original action space.
        initial_stddev: (DEFAULT: 1e-7) initial standard deviation in the original action space.
    """

    def __init__(
        self,
        initial_mean: float = 1e-6,
        initial_stddev: float = 1e-7,
    ):
        # Store init params - after projecting them to the internal "log" space.
        self._initial_mean = torch.Tensor([initial_mean])
        # TODO: think: currently this maps only to internal log space, not the desired log(log) space of stddev.
        self._initial_logstddev = torch.Tensor([log(initial_stddev)])

        # Store policy params - after projecting them to the internal "log" space.
        self._mean = nn.parameter.Parameter(torch.Tensor([initial_mean]))
        # TODO: think: currently this maps only to internal log space, not the desired log(log) space of stddev.
        self._logstddev = nn.parameter.Parameter(torch.Tensor([log(initial_stddev)]))

    def mean(self, initial: bool = False):
        """Returns:
        Mean in the internal action (bid) space.
        """
        if initial:
            return self._initial_mean.clamp_max(1e-1)
        else:
            return self._mean.clamp_max(1e-1)

    def stddev(self, initial: bool = False):
        """Returns:
        Std dev in the internal action (bid) space.
        """
        # TODO: rething the order here -> clamp self.stddev?
        if initial:
            return self._initial_logstddev.exp().clamp_max(1e-2)
        else:
            return self._logstddev.exp().clamp_max(1e-2)

    @property
    def params(self):
        """Returns:
        List of trainable parameters.
        """
        return [self._mean, self._logstddev]

    def distribution(self) -> torch.distributions.Normal:
        """Returns:
        Distribution in the internal space.
        """
        # TODO: rething the order here -> clamp self.stddev?
        return distributions.Normal(self.mean(), self.stddev())

    def get_bids(self):
        """Samples action from the action space, add it to action buffer and returns it.

        Return:
            Action sampled from the action space.
        """
        # Sample action from distribution.
        dist = self.distribution()
        action = dist.rsample().detach().item()
        # assert isinstance(action, float)

        # Add action to buffer (TODO: (re)think the decoupled buffer-scaling logic)
        if hasattr(self, "action_buffer"):
            self.action_buffer.append(action)

        return action

    def get_action(self):
        """Calls get_bids() and scale() to return scaled value."""
        bid = self.get_bids()
        # TODO: introduce min_multiplier parameter.
        # if bid < 1e-20:
        #    bid = 1e-20
        return bid

    async def generate_plot_data(
        self, min_x: float, max_x: float, num_points: int = 200, logspace: bool = False
    ):
        """Generates action distribution for a given cost multiplier range.

        Args:
            min_x (float): Lower bound cost multiplier.
            max_x (float): Upper bound cost multiplier.
            num_points (int, optional): Number of points. Defaults to 200.

        Returns:
            ([x1, x2, ...], [y1, y2, ...], [iy1, iy2, ...]): Triplet of lists of x, y (current policy PDF) and iy (init policy PDF).
        """

        # Prepare points in the "unscaled" x.
        if logspace:
            agent_x = np.logspace(np.log10(min_x), np.log10(max_x), num_points, base=10)
        else:
            agent_x = np.linspace(min_x, max_x, num_points)

        # Get agent's PDF for points in the "inverted scale".
        policy_mean = self.mean().detach().numpy()
        policy_stddev = self.stddev().detach().numpy()
        policy_y = stats.norm.pdf(agent_x, policy_mean, policy_stddev)
        # Scale to 0.5.
        policy_y = 0.5 * policy_y / max(policy_y)

        # Get agent's init PDF for "unscaled" x.
        init_mean = self.mean(initial=True).detach().numpy()
        init_stddev = self.stddev(initial=True).detach().numpy()
        init_y = stats.norm.pdf(agent_x, init_mean, init_stddev)
        # Scale to 0.5.
        init_y = 0.5 * init_y / max(init_y)

        # Return x, current and init policies.
        return {
            "x": agent_x,
            "policy": policy_y,
            "init policy": init_y,
        }


class DeterministicAction(Action):
    """Mixin class for agents with deterministic actions expressed in the original action (price multiplier) space.
    Mixin provides methods for moving for getting action, visualization etc.

    Args:
        initial_value: (DEFAULT: 1e-6) initial value in the action space.
    """

    def __init__(
        self,
        initial_value: float = 1e-6,
    ):
        # Store init params.
        self._value = initial_value

    @property
    def params(self):
        """Returns:
        List of trainable parameters.
        """
        return None

    def get_bids(self):
        """Gets, adds it to action buffer and returns it.

        Return:
            Action sampled from the action space.
        """
        action = self._value

        # Add action to buffer (TODO: (re)think the decoupled buffer-scaling logic)
        if hasattr(self, "action_buffer"):
            self.action_buffer.append(action)

        return action

    def get_action(self):
        """Calls get_bids() to return value."""
        return self.get_bids()

    async def generate_plot_data(
        self, min_x: float, max_x: float, num_points: int = 200, logspace: bool = False
    ):
        """Generates action distribution for a given cost multiplier range.

        Args:
            min_x (float): Lower bound cost multiplier.
            max_x (float): Upper bound cost multiplier.
            num_points (int, optional): Number of points. Defaults to 200.

        Returns:
            ([x1, x2, ...], [y1, y2, ...], [iy1, iy2, ...]): Triplet of lists of x, y (current policy PDF) and iy (init policy PDF).
        """

        # Prepare points in the "unscaled" x.
        agent_x = [min_x, self._value, self._value, self._value, max_x]

        # Dirac at value.
        policy_y = [0, 0, 0.5, 0, 0]

        # Return x, current and init policies.
        return {
            "x": agent_x,
            "policy": policy_y,
        }
