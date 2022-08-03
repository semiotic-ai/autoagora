# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
from asyncio import run

import matplotlib.animation as animation
import matplotlib.pyplot as plt

from price_multiplier_bandit.agent_factory import AgentFactory, add_agent_argparse
from price_multiplier_bandit.environment_factory import (
    EnvironmentFactory,
    add_environment_argparse,
)


def add_experiment_argparse(parser: argparse.ArgumentParser):
    """Adds argparse arguments related to experiment to parser."""
    parser.add_argument(
        "-i",
        "--iterations",
        default=3500,
        type=int,
        help="Sets the length of the experiment / number of iterations (DEFAULT: 3500)",
    )
    parser.add_argument(
        "-f",
        "--fast-forward-factor",
        default=20,
        type=int,
        help="Sets the fast forward factor (DEFAULT: 50)",
    )
    parser.add_argument(
        "--show", action="store_true", help="If set, shows the animation"
    )
    parser.add_argument(
        "--save", action="store_true", help="If set, saves the animation to a file"
    )


if __name__ == "__main__":
    # Init argparse.
    parser = argparse.ArgumentParser(
        usage="%(prog)s [-a ...] [-e ...] [-i ...] [--show] [--save]",
        description="Runs agent simulation and (optionally) shows it and/or saves it to a file.",
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

    # Generate the filename.
    FILENAME = f"{bandit}_{environment}.mp4"

    fig, ax = plt.subplots()
    container = []

    # Environment x.
    min_x = 1e-10
    max_x = 5e-6

    print(f"Training {bandit} on {environment}. Please wait...")
    for i in range(args.iterations):
        # X. Collect the values for visualization of non-stationary environment: subgraph queries/price plot.
        if i % args.fast_forward_factor == 0:
            # Plot environment.
            env_x, env_y = run(environment.generate_plot_data(min_x, max_x))
            (im_env,) = plt.plot(env_x, env_y, color="grey")

        # 1. Get bid from the agent (action)
        scaled_bid = bandit.get_action()

        # 2. Act: set multiplier in the environment.
        run(environment.set_cost_multiplier(scaled_bid))

        # 3. Get the reward.
        # Get queries per second.
        queries_per_second = run(environment.queries_per_second())
        # Turn it into "monies".
        monies_per_second = queries_per_second * scaled_bid
        # Add reward.
        bandit.add_reward(monies_per_second)

        # 4. Update the policy.
        loss = bandit.update_policy()

        # X. Collect the values for visualization of agent's gaussian policy.
        if i % args.fast_forward_factor == 0:
            agent_x, agent_y, init_agent_y = run(
                bandit.generate_plot_data(min_x, max_x)
            )
            (img_agent,) = plt.plot(agent_x, agent_y, color="b")
            (img_init_agent,) = plt.plot(agent_x, init_agent_y, color="g")

            # Put both "images" with labels & title into a container.
            ax.set_xlabel("price multiplier")
            # ax.set_ylabel('queries/s ')
            title = ax.text(
                0.5,
                1.05,
                f"time {i}",
                size=plt.rcParams["axes.titlesize"],
                ha="center",
                transform=ax.transAxes,
            )
            legend = ax.legend([im_env, img_agent, img_init_agent], ["Queries/s", "Policy PDF", "Init Policy PDF"])  # type: ignore
            container.append([im_env, img_agent, title])  # type: ignore

        # 5. Make a step.
        environment.step()

    if args.show or args.save:
        print("Plotting animation...")
        ani = animation.ArtistAnimation(
            fig, container, interval=50, blit=False, repeat_delay=1000
        )

        # Set up formatting for the movie files.
        Writer = animation.writers["ffmpeg"]
        writer = Writer(fps=15, metadata=dict(artist="Me"), bitrate=1800)

        if args.show:
            plt.show()

        if args.save:
            print(f"Saving movie, please wait...")
            ani.save(FILENAME, writer=writer)
            print(f"Movie saved to '{FILENAME}'")
