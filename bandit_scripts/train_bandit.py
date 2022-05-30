# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import numpy as np
from anyio import run
from torch.utils.tensorboard.writer import SummaryWriter

from price_multiplier_bandit.price_bandit import (
    ProximalPolicyOptimizationBandit,
    RollingMemContinuousBandit,
    VanillaPolicyGradientBandit,
)
from price_multiplier_bandit.simulated_subgraph import (
    NoisyCyclicQueriesSubgraph,
    NoisyQueriesSubgraph,
)

ITERATIONS = 3500
BATCH_SIZE = 10

try:
    import socket
    from datetime import datetime

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    log_dir = os.path.join(
        "runs", current_time + "_" + socket.gethostname() + sys.argv[1]
    )

    writer = SummaryWriter(log_dir)
except:
    writer = SummaryWriter()


async def main_loop():
    vars = {"GLOBAL_COST_MULTIPLIER": 0.000001, "DEFAULT_COST": 50}

    # Instantiate environment.
    environment = NoisyCyclicQueriesSubgraph()

    # agora_model = await model_builder(SUBGRAPH)
    # await set_cost_model(SUBGRAPH, agora_model)

    bandit = RollingMemContinuousBandit(
        learning_rate=0.01,
        buffer_max_size=BATCH_SIZE,
    )

    total_money = 0

    print("Training agent. Please wait...")
    for i in range(ITERATIONS):
        writer.add_scalar("Distribution mean", bandit.mean, i)
        writer.add_scalar("Distribution stddev", bandit.logstddev.exp(), i)

        # 1. Get bid from the agent (action)
        scaled_bid = bandit.get_action()
        writer.add_scalar("Price multiplier", scaled_bid, i)

        # 2. Act: set multiplier in the environment.
        await environment.set_cost_multiplier(scaled_bid)

        # 3. Get the reward.
        # Get queries per second.
        queries_per_second = await environment.queries_per_second()
        writer.add_scalar("Queries per second", queries_per_second, i)

        # Turn it into "monies".
        monies_per_second = queries_per_second * scaled_bid
        writer.add_scalar("Monies per second", monies_per_second, i)
        total_money += monies_per_second
        writer.add_scalar("Monies", total_money, i)

        # Add reward.
        bandit.add_reward(monies_per_second)

        # 4. Update the policy.
        loss = bandit.update_policy()
        if loss is not None:
            writer.add_scalar("Loss", loss, i)

        # 5. Make a step.
        environment.step()

    print(f"Training finished. Logs saved to '{writer.log_dir}'")


if __name__ == "__main__":
    run(main_loop)
