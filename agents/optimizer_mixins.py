# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch.optim


class Optimizer(object):
    """Symbolic optimizer mixin class.

    Args:
        optimizer: optimizer to be mixed into the agent's class.
    """

    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
