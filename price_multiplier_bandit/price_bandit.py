# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from math import exp, log
import numpy as np
import scipy.stats as stats
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
        # Store init params.
        self._initial_mean = torch.Tensor([initial_mean])
        self._initial_logstddev = torch.Tensor([initial_logstddev])

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
        #print("self.reward_buffer = ", self.reward_buffer)
        #print(
        #    "self.mean = ",
        #    self.mean.detach(),
        #    " self.logstddev = ",
        #    self.logstddev.detach(),
        #)
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
            try:
                #print(f"x = {x}  => exp(x) * 1e-6 = {exp(x) * 1e-6}")
                return exp(x) * 1e-6
            except OverflowError:
                #print(f"!! OverflowError in exp(x) * 1e-6 for x = {x}!!")
                exit(-1)
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

    async def generate_plot_data(self, min_x: float, max_x: float, num_points: int = 200):
        """Generates action distribution for a given cost multiplier range.

        Args:
            min_x (float): Lower bound cost multiplier.
            max_x (float): Upper bound cost multiplier.
            num_points (int, optional): Number of points. Defaults to 200.

        Returns:
            ([x1, x2, ...], [y1, y2, ...], [iy1, iy2, ...]): Triplet of lists of x, y (current policy PDF) and iy (init policy PDF).
        """

        # Rescale x.
        #agent_min_x = self.inv_scale(min_x)
        #agent_max_x = self.inv_scale(max_x)
        
        # Prepare "scaled" and "unscaled" x.
        #agent_x = np.linspace(agent_min_x, agent_max_x, 200)
        #agent_x_scaled = [self.scale(x) for x in agent_x]

        agent_x_scaled = np.linspace(min_x, max_x, 200)
        agent_x = [self.inv_scale(x) for x in agent_x_scaled]

        # Get agent's PDF for "unscaled" x.
        policy_mean = self.mean.detach().numpy()
        policy_stddev = self.logstddev.exp().detach().numpy()
        policy_y = stats.norm.pdf(agent_x, policy_mean, policy_stddev) * policy_stddev

        # Get agent's init PDF for "unscaled" x.
        init_mean = self._initial_mean.detach().numpy()
        init_stddev = self._initial_logstddev.exp().detach().numpy()
        init_y = stats.norm.pdf(agent_x, init_mean, init_stddev) * init_stddev

        # Return x, y and iy.
        return agent_x_scaled, policy_y, init_y


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
        dist = distributions.Normal(self.mean, self.logstddev.exp())

        if orig_log_prob is None:
            orig_log_prob = dist.log_prob(torch.Tensor(self.action_buffer)).detach()
        else:
            orig_log_prob = torch.Tensor(orig_log_prob)

        # KL loss used for mean and logstddev.
        kl_loss_fn = torch.nn.KLDivLoss()

        for _ in range(self.ppo_iterations):
            # Get log prob of bids coming from normal distribution
            dist = distributions.Normal(self.mean, self.logstddev.exp())

            new_log_prob = dist.log_prob(torch.Tensor(self.action_buffer))

            ratio = torch.exp(new_log_prob - orig_log_prob)

            ppo_loss = -torch.min(
                ratio * advantage,
                torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage,
            )
            entropy_loss = -dist.entropy()

            # Calculate KL losses.
            #kl_loss_logstd = -min(
            #    abs(kl_loss_fn(self.logstddev, self._initial_logstddev)), 1e-1
            #)
            #kl_loss_mean = -min(abs(kl_loss_fn(self.mean, self._initial_mean)), 1e-3)

            # Calculate the final loss.
            loss = (
                ppo_loss
                + self.entropy_coeff * entropy_loss
                #+ kl_loss_mean
                #+ kl_loss_logstd
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
