# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from math import exp, log
from multiprocessing.sharedctypes import Value
from typing import Union, overload

import torch
from torch import distributions, nn

from price_multiplier_bandit.agent import Agent


class ContinuousActionBandit(Agent):
    """Abstract bandit class with continuous action space represented as a gausian.
    The agent internally stores and manages its own experience reply buffer with past actions and rewards.

    Args:
        learning_rate: learning rate.
        initial_mean: (DEFAULT: 0.0) initial mean.
        initial_logstddev: (DEFAULT: 0.4) initial (log) standard deviation.
        buffer_max_size: (DEFAULT: 30) indicates the maximum size of buffer. If buffer_max_size>0, then the buffer will be truncated to this size.

    """

    def __init__(
        self,
        learning_rate: float,
        initial_mean: float = 0.0,
        initial_logstddev: float = 0.4,
        buffer_max_size: int = 30,
    ):
        self.initial_mean = initial_mean
        self.initial_logstddev = initial_logstddev

        # Store policy params.
        self.mean = nn.parameter.Parameter(torch.Tensor([initial_mean]))
        self.logstddev = nn.parameter.Parameter(torch.Tensor([initial_logstddev]))

        # Experience reply buffer.
        self.buffer_max_size = buffer_max_size
        self.action_buffer = []
        self.reward_buffer = []

        # Initialize optimizer.
        self.optimizer = torch.optim.Adam([self.mean, self.logstddev], lr=learning_rate)
        self.learning_rate = learning_rate

    def reset(self):
        self.mean = nn.parameter.Parameter(torch.Tensor([self.initial_mean]))
        self.logstddev = nn.parameter.Parameter(torch.Tensor([self.initial_logstddev]))
        self.action_buffer = []
        self.reward_buffer = []
        self.optimizer = torch.optim.Adam(
            [self.mean, self.logstddev], lr=self.learning_rate
        )

    def __str__(self):
        """
        Return:
            String describing the class and highlighting of its main params.
        """
        return f"{self.__class__.__name__}(buffer_size={self.buffer_max_size}.learning_rate={self.learning_rate})"

    def get_bids(self):
        """Samples action from the action space, add it to action buffer and returns it.

        Return:
            Action sampled from the action space.
        """
        # Sample action from distribution.
        dist = distributions.Normal(self.mean.detach(), self.logstddev.detach().exp())
        action = dist.rsample().item()
        assert isinstance(action, float)

        # Add action to buffer.
        self.action_buffer.append(action)

        return action

    def get_action(self):
        """Calls get_bids() and scale() to return scaled value."""
        bid = self.get_bids()
        scaled_bid = self.scale(bid)
        return scaled_bid

    @overload
    @staticmethod
    def scale(x: float) -> float:
        ...

    @overload
    @staticmethod
    def scale(x: torch.Tensor) -> torch.Tensor:
        ...

    @staticmethod
    def scale(x: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """Scales the value."""
        if isinstance(x, float):
            return exp(x) * 1e-6
        elif isinstance(x, torch.Tensor):
            return x.exp() * 1e-6
        else:
            raise TypeError(f"Invalid type '{type(x)}'")

    @overload
    @staticmethod
    def inv_scale(x: float) -> float:
        ...

    @overload
    @staticmethod
    def inv_scale(x: torch.Tensor) -> torch.Tensor:
        ...

    @staticmethod
    def inv_scale(x: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """Inverse operation to value scaling."""
        if isinstance(x, float):
            return log(x * 1e6)
        elif isinstance(x, torch.Tensor):
            return (x * 1e6).log()
        else:
            raise TypeError(f"Invalid type '{type(x)}'")

    def add_reward(self, reward):
        """Adds reward to the buffer.

        Args:
            reward: reward to be added.
        """
        self.reward_buffer.append(reward)

    def validate_experience_buffer(self):
        """Validates whether both buffers have the same size.

        Raises:
            ValueError if lengths of action and reward buffers are different.
        """
        if len(self.action_buffer) != len(self.reward_buffer):
            raise ValueError("Action and reward buffers need to be of the same size!")

    def is_experience_buffer_full(self):
        """
        Return:
            (True/False) informing whether the buffer is full.
        """
        # Check if buffer is full.
        if len(self.action_buffer) == self.buffer_max_size:
            return True
        else:
            return False

    def truncate_experience_buffer(self, buffer_max_size=None):
        """Truncates buffer size."""
        # Get max buffer size.
        if buffer_max_size is None:
            buffer_max_size = self.buffer_max_size

        # Truncate only if needed.
        if self.buffer_max_size > 0:
            while len(self.action_buffer) > self.buffer_max_size:
                self.action_buffer.pop(0)
                self.reward_buffer.pop(0)

    def clear_experience_buffer(self):
        """Clears the experience buffer."""
        self.action_buffer = []
        self.reward_buffer = []


class VanillaPolicyGradientBandit(ContinuousActionBandit):
    """Bandit with continuous action space using vanilla policy gradients to optimize its policy."""

    def update_policy(self):
        """Updates agent policy using vanilla policy gradients."""
        # Validate buffer.
        self.validate_experience_buffer()

        # Truncate if needed.
        self.truncate_experience_buffer()

        # Train only if the buffer is full.
        if not self.is_experience_buffer_full():
            return

        # Standardize if using batches of data.
        rewards = torch.Tensor(self.reward_buffer)
        if len(self.reward_buffer) > 1:
            advantage = torch.Tensor(
                (rewards - rewards.mean()) / (rewards.std() + 1e-10)
            )
        else:
            advantage = rewards

        # Get log prob of bids coming from normal distribution
        dist = distributions.Normal(self.mean, self.logstddev.exp())
        log_prob = dist.log_prob(torch.Tensor(self.action_buffer))

        # Calcualte loss.
        loss = (-log_prob * advantage).mean() + torch.exp(-self.logstddev - 5)

        # Optimize model params.
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # We have "used" the samples for training - clear the buffer.
        self.clear_experience_buffer()

        # Return loss.
        return loss.item()


class ProximalPolicyOptimizationBandit(ContinuousActionBandit):
    """Bandit with continuous action space using proximal policy optimization.
    The agent internally stores and manages its own experience reply buffer with past actions and rewards.

    Args:
        learning_rate: learning rate.
        initial_mean: (DEFAULT: 0.0) initial mean.
        initial_logstddev: (DEFAULT: 0.4) initial (log) standard deviation.
        buffer_max_size: (DEFAULT: 30) indicates the maximum size of buffer. If buffer_max_size>0, then the buffer will be truncated to this size.
        eps_clip: (DEFAULT: 0.1) epsilon used in PPO clipping.
        ppo_iterations: (DEFAULT: 50) number of optimization steps.
        entropy_coeff: (DEFAULT: 1e-1) entropy coefficient for the loss calculation.
    """

    def __init__(
        self,
        learning_rate: float,
        initial_mean: float = 0.0,
        initial_logstddev: float = 0.4,
        buffer_max_size: int = 30,
        eps_clip: float = 0.1,
        ppo_iterations: int = 10,
        entropy_coeff: float = 1e-1,
    ):
        # Call parent class constructor.
        super().__init__(
            learning_rate=learning_rate,
            initial_mean=initial_mean,
            initial_logstddev=initial_logstddev,
            buffer_max_size=buffer_max_size,
        )

        # Memorize PPO-related variables.
        self.eps_clip = eps_clip
        self.ppo_iterations = ppo_iterations
        self.entropy_coeff = entropy_coeff

    def __str__(self):
        """
        Return:
            String describing the class and highlighting of its main params.
        """
        return f"{self.__class__.__name__}(buffer_size={self.buffer_max_size}.learning_rate={self.learning_rate}.ppo_iterations={self.ppo_iterations})"

    def ppo_update(self, orig_log_prob=None):
        """Implements proximal policy update."""
        # Standardize if using batches of data.
        rewards = torch.Tensor(self.reward_buffer)
        if len(self.reward_buffer) > 1:
            advantage = torch.Tensor(
                (rewards - rewards.mean()) / (rewards.std() + 1e-10)
            )
        else:
            advantage = rewards

        # Get log prob of bids coming from normal distribution
        dist = distributions.Normal(self.mean, self.logstddev.exp())

        if orig_log_prob is None:
            orig_log_prob = dist.log_prob(torch.Tensor(self.action_buffer)).detach()
        else:
            orig_log_prob = torch.Tensor(orig_log_prob)

        for i in range(self.ppo_iterations):
            dist = distributions.Normal(self.mean, self.logstddev.exp())

            new_log_prob = dist.log_prob(torch.Tensor(self.action_buffer))

            ratio = torch.exp(new_log_prob - orig_log_prob)

            ppo_loss = -torch.min(
                ratio * advantage,
                torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage,
            )
            entropy_loss = -dist.entropy()

            loss = ppo_loss + self.entropy_coeff * entropy_loss

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        return loss.mean().item()  # type: ignore

    def update_policy(self):
        """Updates agent policy using PPO."""

        # Validate buffer.
        self.validate_experience_buffer()

        # Truncate if needed.
        self.truncate_experience_buffer()

        # Train only if the buffer is full.
        if not self.is_experience_buffer_full():
            return

        # Update the policy using PPO.
        loss = self.ppo_update()

        # We have "used" the samples for training - clear the buffer.
        self.clear_experience_buffer()

        # Return loss.
        return loss


class RollingMemContinuousBandit(ProximalPolicyOptimizationBandit):
    """Bandit with continuous action space using proximal policy optimization with a "rolling" experience buffer.
    The agent internally stores and manages its own experience reply buffer with past actions and rewards.

    Args:
        learning_rate: learning rate.
        initial_mean: (DEFAULT: 0.0) initial mean.
        initial_logstddev: (DEFAULT: 0.4) initial (log) standard deviation.
        buffer_max_size: (DEFAULT: 30) indicates the maximum size of buffer. If buffer_max_size>0, then the buffer will be truncated to this size.
        eps_clip: (DEFAULT: 0.1) epsilon used in PPO clipping.
        ppo_iterations: (DEFAULT: 50) number of optimization steps.
        entropy_coeff: (DEFAULT: 1e-1) entropy coefficient for the loss calculation.
    """

    def __init__(
        self,
        learning_rate,
        initial_mean: float = 2.0,
        initial_logstddev: float = 0.4,
        buffer_max_size: int = 10,
        eps_clip: float = 0.1,
        ppo_iterations: int = 10,
        entropy_coeff: float = 1e-1,
    ):
        # Call parent class constructor.
        super().__init__(
            learning_rate=learning_rate,
            initial_mean=initial_mean,
            initial_logstddev=initial_logstddev,
            buffer_max_size=buffer_max_size,
            eps_clip=eps_clip,
            ppo_iterations=ppo_iterations,
            entropy_coeff=entropy_coeff,
        )

        # New buffer for action log probs.
        self.orig_log_prob_buffer = []

    def reset(self):
        super().reset()
        self.orig_log_prob_buffer = []

    def get_bids(self):
        """Samples action from the action space, add it to action buffer and returns it.

        Return:
            Action sampled from the action space.
        """
        # Call parent class method.
        action = super().get_bids()

        # Sample log_prob from distribution.
        dist = distributions.Normal(self.mean.detach(), self.logstddev.detach().exp())
        orig_log_prob = dist.log_prob(torch.Tensor([action])).detach().item()

        # Add to buffer.
        self.orig_log_prob_buffer.append(orig_log_prob)

        return action

    def validate_experience_buffer(self):
        """Validates whether all three buffers have the same size.

        Raises:
            ValueError if lengths of action, reward and action log_prob buffers are different.
        """
        if len(self.action_buffer) != len(self.reward_buffer) or len(
            self.action_buffer
        ) != len(self.orig_log_prob_buffer):
            raise ValueError("Action and reward buffers need to be of the same size!")

    def truncate_experience_buffer(self, buffer_max_size=None):
        """Truncates buffer size."""
        # Get max buffer size.
        if buffer_max_size is None:
            buffer_max_size = self.buffer_max_size

        # Truncate only if needed.
        if self.buffer_max_size > 0:
            while len(self.action_buffer) > self.buffer_max_size:
                self.action_buffer.pop(0)
                self.reward_buffer.pop(0)
                self.orig_log_prob_buffer.pop(0)

    def clear_experience_buffer(self):
        """Clears the experience buffer."""
        self.action_buffer = []
        self.reward_buffer = []
        self.orig_log_prob_buffer = []

    def update_policy(self):
        """Updates agent policy using PPO with rolling buffer (i.e. without clearing the buffer after optimization)."""

        # Validate buffer.
        self.validate_experience_buffer()

        # Truncate if needed.
        self.truncate_experience_buffer()

        # Train only if the buffer is full.
        if not self.is_experience_buffer_full():
            return

        # Update the policy using PPO.
        loss = self.ppo_update(self.orig_log_prob_buffer)

        # Return loss.
        return loss


class SafeRollingMemContinuousBandit(RollingMemContinuousBandit):
    def __init__(
        self,
        learning_rate: float,
        fallback_price_multiplier: float = 1e-6,
        initial_mean: float = 2,
        initial_logstddev: float = 0.4,
        buffer_max_size: int = 10,
        eps_clip: float = 0.1,
        ppo_iterations: int = 10,
        entropy_coeff: float = 0.1,
    ):
        super().__init__(
            learning_rate,
            initial_mean,
            initial_logstddev,
            buffer_max_size,
            eps_clip,
            ppo_iterations,
            entropy_coeff,
        )

        self.fallback_mode = False
        self.fallback_price_multiplier = fallback_price_multiplier

    def reset(self):
        super().reset()
        self.fallback_mode = False

    def is_reward_buffer_zeros(self) -> bool:
        """Wether the reward buffer is full of zeros.

        Returns:
            bool: True if the reward buffer is full of zeros.
        """

        return all(map(lambda x: x == 0.0, self.reward_buffer))

    def is_reward_buffer_never_zero(self) -> bool:
        """Wether the reward buffer values are always greater than zero.

        Returns:
            bool: True if the reward buffer does not contain zero.
        """

        return not any(map(lambda x: x == 0, self.reward_buffer))

    def update_policy(self):
        """Updates agent policy using PPO with rolling buffer (i.e. without clearing the buffer after optimization)."""

        if self.fallback_mode:
            if self.is_reward_buffer_never_zero():
                # Restart the bandit
                self.reset()
            else:
                return

        # Validate buffer.
        self.validate_experience_buffer()

        # Truncate if needed.
        self.truncate_experience_buffer()

        # Train only if the buffer is full and not full of zeros.
        if not self.is_experience_buffer_full():
            return

        if self.is_reward_buffer_zeros():
            self.fallback_mode = True
            return

        # Update the policy using PPO.
        loss = self.ppo_update(self.orig_log_prob_buffer)

        # Return loss.
        return loss

    def get_bids(self):
        if self.fallback_mode:
            return self.fallback_price_multiplier

        return super().get_bids()
