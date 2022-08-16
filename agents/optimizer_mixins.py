# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from torch.optim import Optimizer


class OptimizerMixin(object):
    """Symbolic optimizer mixin class.

    Args:
        optimizer: optimizer to be mixed into the agent's class.
    """

    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
