# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from attr import has

from agents.continuous_action_agent import ContinuousActionBandit


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

        # Turn rewards into tensor.
        rewards = torch.Tensor(self.reward_buffer)

        # Calculate advantage.
        if len(self.reward_buffer) > 1:
            advantage = torch.Tensor(
                (rewards - rewards.mean()) / (rewards.std() + 1e-10)
            )
        else:
            advantage = rewards

        # Get log prob of bids coming from normal distribution
        dist = self.distribution()
        log_prob = dist.log_prob(torch.Tensor(self.action_buffer))

        # Calcualte loss.
        loss = (-log_prob * advantage).mean() + torch.exp(-self._logstddev - 5)

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
        initial_mean: (DEFAULT: 1e-6) initial mean in the original action (i.e. scaled bid) space.
        initial_stddev: (DEFAULT: 1e-7) initial standard deviation in the original action (i.e. scaled bid) space.
        buffer_max_size: (DEFAULT: 30) indicates the maximum size of buffer. If buffer_max_size>0, then the buffer will be truncated to this size.
        eps_clip: (DEFAULT: 0.1) epsilon used in PPO clipping.
        ppo_iterations: (DEFAULT: 50) number of optimization steps.
        entropy_coeff: (DEFAULT: 1e-1) entropy coefficient for the loss calculation.
    """

    def __init__(
        self,
        learning_rate: float,
        initial_mean: float = 1e-6,
        initial_stddev: float = 1e-7,
        buffer_max_size: int = 30,
        eps_clip: float = 0.1,
        ppo_iterations: int = 10,
        entropy_coeff: float = 1e-1,
    ):
        # Call parent class constructor.
        ContinuousActionBandit.__init__(
            self,
            learning_rate=learning_rate,
            initial_mean=initial_mean,
            initial_stddev=initial_stddev,
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
        # Turn rewards into tensor.
        rewards = torch.Tensor(self.reward_buffer)

        # Calculate advantage.
        if len(self.reward_buffer) > 1:
            advantage = torch.Tensor(
                (rewards - rewards.mean()) / (rewards.std() + 1e-10)
            )
        else:
            advantage = rewards

        # Get log prob of bids coming from normal distribution
        dist = self.distribution()

        if orig_log_prob is None:
            orig_log_prob = dist.log_prob(torch.Tensor(self.action_buffer)).detach()
        else:
            orig_log_prob = torch.Tensor(orig_log_prob).detach()

        for _ in range(self.ppo_iterations):
            # Get log prob of bids coming from normal distribution
            dist = self.distribution()

            new_log_prob = dist.log_prob(torch.Tensor(self.action_buffer))

            ratio = torch.exp(new_log_prob - orig_log_prob)

            ppo_loss = -torch.min(
                ratio * advantage,
                torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage,
            )
            entropy_loss = -dist.entropy()

            # Basic PPO loss with entropy.
            loss = ppo_loss + self.entropy_coeff * entropy_loss

            # Graceful fallback pull towards init mean.
            if hasattr(self, "_mean") and hasattr(self, "_initial_mean"):
                loss += torch.nn.L1Loss()(self._mean, self._initial_mean) * 1e-3

            # Graceful fallback pull towards init std dev.
            if hasattr(self, "_logstddev") and hasattr(self, "_initial_logstddev"):
                loss += (
                    torch.nn.L1Loss()(self._logstddev, self._initial_logstddev) * 1e-1
                )

            # Optimize the model parameters.
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
        initial_mean: (DEFAULT: 1e-6) initial mean in the original action (i.e. scaled bid) space.
        initial_stddev: (DEFAULT: 1e-7) initial standard deviation in the original action (i.e. scaled bid) space.
        buffer_max_size: (DEFAULT: 30) indicates the maximum size of buffer. If buffer_max_size>0, then the buffer will be truncated to this size.
        eps_clip: (DEFAULT: 0.1) epsilon used in PPO clipping.
        ppo_iterations: (DEFAULT: 50) number of optimization steps.
        entropy_coeff: (DEFAULT: 1e-1) entropy coefficient for the loss calculation.
    """

    def __init__(
        self,
        learning_rate,
        initial_mean: float = 1e-6,
        initial_stddev: float = 1e-7,
        buffer_max_size: int = 10,
        eps_clip: float = 0.1,
        ppo_iterations: int = 10,
        entropy_coeff: float = 1e-1,
    ):
        # Call parent class constructor.
        ProximalPolicyOptimizationBandit.__init__(
            self,
            learning_rate=learning_rate,
            initial_mean=initial_mean,
            initial_stddev=initial_stddev,
            buffer_max_size=buffer_max_size,
            eps_clip=eps_clip,
            ppo_iterations=ppo_iterations,
            entropy_coeff=entropy_coeff,
        )

        # New buffer for action log probs.
        self.orig_log_prob_buffer = []

    def get_bids(self):
        """Samples action from the action space, add it to action buffer and returns it.

        Return:
            Action sampled from the action space.
        """
        # Call parent class method.
        action = super().get_bids()

        # Sample log_prob from distribution.
        dist = self.distribution()
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
