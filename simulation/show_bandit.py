# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
from asyncio import run
from typing import List

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.artist import Artist

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
        usage="%(prog)s [-c ...] [-i ...] [--show] [--save]",
        description="Runs single-agent simulation and (optionally) shows it and/or saves it to a file.",
    )
    add_experiment_argparse(parser=parser)

    # Initialize the simulation.
    args, environment, agents = init_simulation(parser=parser)
    # We need the environment to be SimulatedSubgraph
    assert isinstance(environment, SimulatedSubgraph)
    (agent_name, bandit) = next(iter(agents.items()))

    # Generate the filename.
    FILENAME = f"{args.config}.mp4"

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
        else:  # Avoid unbound variables
            im_env = None

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
            # Containers for frame.
            assert im_env  # im_env shouldn't be None
            frame_image_container: List["Artist"] = [im_env]
            frame_legend_container = ["Queries/s"]

            # Get data.
            data = run(bandit.generate_plot_data(min_x, max_x))
            agent_x = data.pop("x")
            agent_y = data["policy"]

            # Plot policy and add it to last list in container.
            (img_agent,) = plt.plot(agent_x, agent_y, color="b")
            frame_image_container.append(img_agent)
            frame_legend_container.append(f"Agent {agent_name}: policy")

            # Plot init policy and add it to last list in container.
            if "init policy" in data.keys():
                init_agent_y = data["init policy"]
                (img_init_agent,) = plt.plot(
                    agent_x, init_agent_y, color="b", linestyle="dashed"
                )
                frame_image_container.append(img_init_agent)
                frame_legend_container.append(f"Agent {agent_name}: init policy")

            # Plot agent q/s.
            agent_qps_x = min(max_x, max(min_x, scaled_bid))
            img_agent_qps = plt.scatter(
                [agent_qps_x],
                [queries_per_second],
                marker="o",
                color="b",
            )
            frame_image_container.append(img_agent_qps)
            frame_legend_container.append(f"Agent {agent_name}: action => q/s")

            # Put both "images" with labels & title into a container.
            ax.set_xlabel("Price multiplier")  # type: ignore
            # ax.set_ylabel('queries/s ')
            title = ax.text(  # type: ignore
                0.5,
                1.05,
                f"time {i}",
                size=plt.rcParams["axes.titlesize"],
                ha="center",
                transform=ax.transAxes,  # type: ignore
            )
            legend = ax.legend(frame_image_container, frame_legend_container)  # type: ignore
            container.append([*frame_image_container, title])

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
