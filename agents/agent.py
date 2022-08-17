# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from agents.action_mixins import Action
from agents.policy_mixins import Policy


class Agent(Action, Policy):
    """Abstract agent class defining agent's elementary interface by composing action and policy mixins."""

    def __init__(self, name: str):
        self._name = name
        # No default optimizer.
        self._optimizer = None

    @property
    def name(self):
        """Returns:
        Agent name.
        """
        return self._name

    @property
    def optimizer(self):
        """Returns:
        Agent optimizer.
        """
        if self._optimizer is None:
            raise ValueError("Optimizer not set!")
        return self._optimizer

    def __str__(self):
        """
        Return:
            String with agent's name and class.
        """
        return f"{self._name}({self.__class__.__name__})"
