# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import sys

import numpy as np
from anyio import run
from torch.utils.tensorboard.writer import SummaryWriter

from price_multiplier_bandit.agent_factory import AgentFactory, add_agent_argparse
from price_multiplier_bandit.environment_factory import (
    EnvironmentFactory,
    add_environment_argparse,
)


def add_experiment_argparse(parser: argparse):
    """Adds argparse arguments related to experiment to parser."""
    parser.add_argument(
        "-i",
        "--iterations",
        default=3500,
        type=int,
        help="Sets the length of the experiment / number of args.iterations (DEFAULT: 3000)",
    )


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

    # Init argparse.
    parser = argparse.ArgumentParser(
        usage="%(prog)s [-a ...] [-e ...] [-n ...]",
        description="Trains an agent on a given environment.",
    )
    add_experiment_argparse(parser=parser)
    add_agent_argparse(parser=parser)
    add_environment_argparse(parser=parser)
    # Parse arguments
    args = parser.parse_args()

    # Instantiate the agent.
    bandit = AgentFactory(
        agent_type=args.agent,
        learning_rate=args.learning_rate,
        buffer_max_size=args.buffer_size,
    )

    # Instantiate the environment.
    environment = EnvironmentFactory(environment_type=args.environment)

    total_money = 0
    print(f"Training {bandit} on {environment}. Please wait...")
    for i in range(args.iterations):
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
