# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
from asyncio import run
from re import A

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import colors

from environments.simulated_subgraph import SimulatedSubgraph
from simulation.controller import init_simulation
from simulation.show_bandit import add_experiment_argparse

logging.basicConfig(level="WARN", format="%(message)s")

    # Init argparse.
    parser = argparse.ArgumentParser(
        usage="%(prog)s [-c ...] [-i ...] [--show] [--save]",
        description="Runs multi-agent simulation and (optionally) shows it and/or saves it to a file.",
    )
    add_experiment_argparse(parser=parser)

    # Initialize the simulation.
    args, environment, agents = init_simulation(parser=parser)
    # We need the environment to be SimulatedSubgraph
    assert isinstance(environment, SimulatedSubgraph)

    # Generate the filename.
    FILENAME = f"{args.config}.mp4"

    fig, ax = plt.subplots()

    # Environment x.
    min_x = 1e-10
    max_x = 5e-6

    # Create containers.
    image_container = []
    legend_container = []
    agent_colors = [
        colors.to_rgba(c) for c in plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ]

    print(f"Training {len(agents)} x agents on {environment}. Please wait...")
    for i in range(args.iterations):
        logging.debug("=" * 20 + " step %s " + "=" * 20, i)

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
        for agent_id, (agent_name, agent) in enumerate(agents.items()):
            # 1. Get bid from the agent (action)
            scaled_bids.append(agent.get_action())
            if agent_id == 0:
                logging.debug("Agent %s action: %s", agent_id, scaled_bids[agent_id])

            # 2. Act: set multiplier in the environment.
            run(
                environment.set_cost_multiplier(
                    scaled_bids[agent_id], agent_id=agent_id
                )
            )

        # Get observations for all agents.
        queries_per_second = []
        for agent_id, (agent_name, agent) in enumerate(agents.items()):
            # 3. Get the rewards.
            # Get queries per second for a given .
            queries_per_second.append(
                run(environment.queries_per_second(agent_id=agent_id))
            )
            # Turn it into "monies".
            monies_per_second = queries_per_second[agent_id] * scaled_bids[agent_id]
            # Add reward.
            agent.add_reward(monies_per_second)

            # 4. Update the policy.
            if True:  # agent_id == 0:
                if hasattr(agent, "reward_buffer"):
                    logging.debug(
                        "Agent %s reward_buffer = %s",
                        agent_id,
                        agent.reward_buffer,
                    )
                    logging.debug(
                        "Agent %s action_buffer = %s",
                        agent_id,
                        agent.action_buffer,
                    )
                if hasattr(agent, "mean"):
                    logging.debug(
                        "Agent %s mean = %s",
                        agent_id,
                        agent.mean(),
                    )
                    logging.debug(
                        f"Agent %s stddev = %s",
                        agent_id,
                        agent.stddev(),
                    )
                    logging.debug(
                        f"Agent %s initial_mean = %s",
                        agent_id,
                        agent.mean(initial=True),
                    )

                logging.debug(
                    "Agent %s observation: %s",
                    agent_id,
                    queries_per_second[agent_id],
                )
            loss = agent.update_policy()
            logging.debug(f"Agent %s loss = %s", agent_id, loss)

        # X. Collect the values for visualization of agent's gaussian policy.
        if i % args.fast_forward_factor == 0:
            for agent_id, (agent_name, agent) in enumerate(agents.items()):

                # Get data.
                data = run(agent.generate_plot_data(min_x, max_x))
                agent_x = data.pop("x")
                agent_y = data["policy"]
                agent_color = agent_colors[agent_id % len(agent_colors)]

                # Plot policy and add it to last list in container.
                (img_agent,) = plt.plot(agent_x, agent_y, color=agent_color)
                image_container[-1].append(img_agent)
                legend_container[-1].append(f"Agent {agent_name}: policy")

                # Plot init policy and add it to last list in container.
                if "init policy" in data.keys():
                    init_agent_y = data["init policy"]
                    (img_init_agent,) = plt.plot(
                        agent_x, init_agent_y, color=agent_color, linestyle="dashed"
                    )
                    image_container[-1].append(img_init_agent)
                    legend_container[-1].append(f"Agent {agent_name}: init policy")

                # Agent q/s.
                agent_qps_x = min(max_x, max(min_x, scaled_bids[agent_id]))
                img_agent_qps = plt.scatter(
                    [agent_qps_x],
                    [queries_per_second[agent_id]],
                    marker="o",
                    color=agent_color,
                )
                image_container[-1].append(img_agent_qps)
                legend_container[-1].append(f"Agent {agent_name}: action => q/s")

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
