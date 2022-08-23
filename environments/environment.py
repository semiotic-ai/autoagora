# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod

from agents.mixin import ABCMixin


class Environment(ABCMixin):
    """Abstract environment class defining an elementary interface"""

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def reset(self, *args, **kwargs):
        """Abstract method for resetting the state of the environment."""
        pass

    @abstractmethod
    def step(self, *args, **kwargs):
        """Abstract method for executing the step."""
        pass

    @abstractmethod
    def observation(self, *args, **kwargs):
        """Abstract method returning observation based on the environment step."""
        pass
