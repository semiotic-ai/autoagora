# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import sys

from anyio import run
from torch.utils.tensorboard.writer import SummaryWriter

from environments.simulated_subgraph import SimulatedSubgraph
from simulation.controller import init_simulation


def add_experiment_argparse(parser: argparse.ArgumentParser):
    """Adds argparse arguments related to experiment to parser."""
    parser.add_argument(
        "-c",
        "--config",
        default="simulation/configs/3different_agents_noisy_cyclic.json",
        type=str,
        help="Sets the config file (DEFAULT: simulation/configs/3different_agents_noisy_cyclic.json)",
    )
    parser.add_argument(
        "-i",
        "--iterations",
        default=3500,
        type=int,
        help="Sets the length of the experiment / number of iterations (DEFAULT: 3500)",
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
        usage="%(prog)s [-c ...] [-i ...]",
        description="Trains an agent on a given environment.",
    )
    add_experiment_argparse(parser=parser)
    # Parse arguments
    args = parser.parse_args()

    # Initialize the simulation.
    args, environment, agents = init_simulation(parser=parser)
    # We need the environment to be SimulatedSubgraph
    assert isinstance(environment, SimulatedSubgraph)

    (_, bandit) = next(iter(agents.items()))

    total_money = 0
    print(f"Training {bandit} on {environment}. Please wait...")
    for i in range(args.iterations):
        writer.add_scalar("Distribution mean", bandit.mean(), i)
        writer.add_scalar("Distribution stddev", bandit.stddev(), i)

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
