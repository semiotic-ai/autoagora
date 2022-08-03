# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
from asyncio import run
from re import A

from matplotlib import colors
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from price_multiplier_bandit.agent_factory import AgentFactory, add_agent_argparse
from price_multiplier_bandit.environment_factory import (
    EnvironmentFactory,
    add_environment_argparse,
)
from bandit_scripts.show_bandit import add_experiment_argparse as single_agent_add_experiment_parse


def add_experiment_argparse(parser: argparse.ArgumentParser):
    """Adds argparse arguments related to experiment to parser."""
    single_agent_add_experiment_parse(parser=parser)
    parser.add_argument(
        "-n",
        "--number",
        default=5,
        type=int,
        help="Sets the number of agents (DEFAULT: 5)",
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
    bandits = [AgentFactory(
            agent_type=args.agent,
            learning_rate=args.learning_rate,
            buffer_max_size=args.buffer_size,
        ) for _ in range(args.number)]

    # Instantiate the environment.
    environment = EnvironmentFactory(environment_type=args.environment)

    # Generate the filename.
    FILENAME = f"{args.number}x{bandits[0]}_{environment}.mp4"

    fig, ax = plt.subplots()

    # Environment x.
    min_x = 1e-10
    max_x = 5e-6

    # Create containers.
    image_container = []
    legend_container = []
    agent_colors = [colors.to_rgba(c) for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]

    print(f"Training {args.number} x {bandits[0]} on {environment}. Please wait...")
    for i in range(args.iterations):
        print("="*20, f" step {i} ", "="*20)

        # X. Visualize the environment.
        if i % args.fast_forward_factor == 0:
            # Plot environment.
            env_x, env_y = run(environment.generate_plot_data(min_x, max_x))
            (im_env,) = plt.plot(env_x, env_y, color="grey")

            # Add new list to image container.
            image_container.append([im_env])
            legend_container.append(["Environment: total q/s"])

        # Execute actions for all agents.
        scaled_bids = []
        for agent_id in range(args.number):
            # 1. Get bid from the agent (action)
            scaled_bids.append(bandits[agent_id].get_action())
            if agent_id == 0:
                print(f"Agent {agent_id} action: ", scaled_bids[agent_id])

            # 2. Act: set multiplier in the environment.
            run(environment.set_cost_multiplier(scaled_bids[agent_id], agent_id=agent_id))

        # Get observations for all agents.
        queries_per_second = []
        for agent_id in range(args.number):
            # 3. Get the rewards.
            # Get queries per second for a given .
            queries_per_second.append(run(environment.queries_per_second(agent_id=agent_id)))
            # Turn it into "monies".
            monies_per_second = queries_per_second[agent_id] * scaled_bids[agent_id]
            # Add reward.
            bandits[agent_id].add_reward(monies_per_second)

            # 4. Update the policy.
            if agent_id == 0:
                print(f"Agent {agent_id} reward_buffer = ", bandits[agent_id].reward_buffer)
                print(
                    f"Agent {agent_id} mean = ",
                    bandits[agent_id].mean.detach(),
                    f"Agent {agent_id} logstddev = ",
                    bandits[agent_id].logstddev.detach(),
                    )

                print(f"Agent {agent_id} observation: ", queries_per_second[agent_id])
            loss = bandits[agent_id].update_policy()


        # X. Collect the values for visualization of agent's gaussian policy.
        if i % args.fast_forward_factor == 0:
            for agent_id in range(args.number):
                agent_x, agent_y, init_agent_y = run(
                    bandits[agent_id].generate_plot_data(min_x, max_x)
                )
                agent_color = agent_colors[agent_id % len(agent_colors)]
                (img_agent,) = plt.plot(agent_x, agent_y, color=agent_color)
                #(img_init_agent,) = plt.plot(agent_x, init_agent_y, color="g")

                # Add image to last list in container.
                image_container[-1].append(img_agent)
                legend_container[-1].append(f"Agent {agent_id}: policy (PDF)")
                
                # Agent q/s.
                agent_qps_x = min(max_x, max(min_x, scaled_bids[agent_id]))
                img_agent_qps = plt.scatter([agent_qps_x], [queries_per_second[agent_id]], marker="o", color=agent_color)
                image_container[-1].append(img_agent_qps)
                legend_container[-1].append(f"Agent {agent_id}: action => q/s")


            # Generate labels & title.
            ax.set_xlabel("Price multiplier")
            # ax.set_ylabel('queries/s ')
            title = ax.text(
                0.5,
                1.05,
                f"time {i}",
                size=plt.rcParams["axes.titlesize"],
                ha="center",
                transform=ax.transAxes,
            )

            # Create legend object.
            legend = ax.legend(image_container[-1], legend_container[-1])  # type: ignore

            # Add title to last list in container.
            image_container[-1].append(title)  # type: ignore

        # 5. Make a step.
        environment.step()

    if args.show or args.save:
        print("Plotting animation...")
        ani = animation.ArtistAnimation(
            fig, image_container, interval=50, blit=False, repeat_delay=1000
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
