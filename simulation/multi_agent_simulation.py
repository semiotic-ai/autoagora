# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
from asyncio import run

import ffmpeg
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from environments.simulated_subgraph import SimulatedSubgraph
from simulation.controller import init_simulation
from simulation.show_bandit import add_experiment_argparse

logging.basicConfig(level="WARN", format="%(message)s")


def main():
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

    ffmpeg_process = None

    # Environment x.
    min_x = 1e-10
    max_x = 5e-6

    # Set up PyQtGraph
    pg.setConfigOption("foreground", "white")
    pg.setConfigOptions(antialias=True)
    app = pg.mkQApp("Plot")
    win = pg.GraphicsLayoutWidget(show=not args.save, title="Multi-agent training")
    win.resize(1000, 800)

    # Create policy plot
    plot_1 = win.addPlot(title="time 0")
    plot_1.setPreferredHeight(600)
    plot_1_legend = plot_1.addLegend(offset=None)
    plot_1_vb = win.addViewBox()  # Empty UI box to contain the legend outside the plot
    plot_1_vb.setFixedWidth(300)
    plot_1_legend.setParentItem(plot_1_vb)
    plot_1.setYRange(0, 1.1)
    plot_1.setXRange(min_x, max_x)
    plot_1.setClipToView(True)
    plot_1.setLabel("left", "Query rate")
    plot_1.setLabel("bottom", "Price multiplier")
    # Policy PD
    agents_dist = [
        plot_1.plot(
            pen=pg.mkPen(color=(i, len(agents) + 1), width=1.5),
            name=f"Agent {agent_name}: policy",
        )
        for i, agent_name in enumerate(agents.keys())
    ]

    # Initial policy PD
    agents_init_dist = [
        plot_1.plot(
            pen=pg.mkPen(color=(i, len(agents) + 1), width=1.5, style=QtCore.Qt.DotLine),  # type: ignore
            name=f"Agent {agent_name}: init policy",
        )
        for i, agent_name in enumerate(agents.keys())
    ]
    # This is a line plot with invisible line and visible data points.
    # Easier to scale with the rest of the plot than with using a ScatterPlot.
    agents_scatter_qps = [
        plot_1.plot(
            pen=pg.mkPen(color=(0, 0, 0, 0), width=0),  # type: ignore
            name=f"Agent {agent_name}: query rate",
            symbolBrush=(i, len(agents) + 1),
            symbolPen="w",
        )
        for i, agent_name in enumerate(agents.keys())
    ]
    # Environment QPS
    env_plot = plot_1.plot(
        pen=pg.mkPen(color="gray", width=1.5), name="Environment: total query rate"
    )

    win.nextRow()

    # Create query volume time plot
    plot_2 = win.addPlot()
    plot_2.setPreferredHeight(200)
    plot_2.setLabel("left", "Query rate")
    plot_2.setLabel("bottom", "Timestep")
    plot_2_legend = plot_2.addLegend(offset=None)
    plot_2_vb = win.addViewBox()
    plot_2_vb.setFixedWidth(300)
    plot_2_legend.setParentItem(plot_2_vb)
    agent_qps_plots = [
        plot_2.plot(
            pen=pg.mkPen(color=(i, len(agents) + 1), width=1.5),
            name=f"Agent {agent_name}",
        )
        for i, agent_name in enumerate(agents.keys())
    ]
    queries_per_second = [[] for _ in agents]

    win.nextRow()

    # Create total queries (un)served plot
    plot_3 = win.addPlot()
    plot_3.setPreferredHeight(200)
    plot_3.setLabel("left", "Total queries")
    plot_3.setLabel("bottom", "Timestep")
    plot_3_legend = plot_3.addLegend(offset=None)
    plot_3_vb = win.addViewBox()
    plot_3_vb.setFixedWidth(300)
    plot_3_legend.setParentItem(plot_3_vb)
    plot_3_legend.anchor((0, 0), (0, 0))

    total_agent_queries_plots = [
        plot_3.plot(
            pen=pg.mkPen(color=(i, len(agents) + 1), width=1.5),
            name=f"Agent {agent_name}",
        )
        for i, agent_name in enumerate(agents.keys())
    ]
    total_unserved_queries_plot = plot_3.plot(
        pen=pg.mkPen(color=(len(agents), len(agents) + 1), width=1.5),
        name=f"Dropped",
    )

    total_agent_queries_data = [[] for _ in agents]
    total_unserved_queries_data = []

    for i in range(args.iterations):
        logging.debug("=" * 20 + " step %s " + "=" * 20, i)

        # X. Visualize the environment.
        if i % args.fast_forward_factor == 0:
            # Plot environment.
            env_x, env_y = run(environment.generate_plot_data(min_x, max_x))
            env_plot.setData(env_x, env_y)

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
        for agent_id, (agent_name, agent) in enumerate(agents.items()):
            # 3. Get the rewards.
            # Get queries per second for a given .
            queries_per_second[agent_id] += [
                run(environment.queries_per_second(agent_id=agent_id))
            ]
            # Turn it into "monies".
            monies_per_second = queries_per_second[agent_id][-1] * scaled_bids[agent_id]
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
                    queries_per_second[agent_id][-1],
                )
            loss = agent.update_policy()
            logging.debug(f"Agent %s loss = %s", agent_id, loss)

            # Agents total queries served (for the plots)
            if i > 0:
                total_agent_queries_data[agent_id] += [
                    total_agent_queries_data[agent_id][-1]
                    + queries_per_second[agent_id][-1]
                ]
            else:
                total_agent_queries_data[agent_id] += [queries_per_second[agent_id][-1]]

        # Total unserved queries
        if environment.__class__.__name__ == "IsaSubgraph":
            if i > 0:
                total_unserved_queries_data += [
                    total_unserved_queries_data[-1]
                    + 1
                    - sum(e[-1] for e in queries_per_second)
                ]
            else:
                total_unserved_queries_data += [
                    1 - sum(e[-1] for e in queries_per_second)
                ]

        # X. Collect the values for visualization of agent's gaussian policy.
        if i % args.fast_forward_factor == 0:
            for agent_id, (agent_name, agent) in enumerate(agents.items()):

                # Get data.
                data = run(agent.generate_plot_data(min_x, max_x))
                agent_x = data.pop("x")
                agent_y = data["policy"]
                agents_dist[agent_id].setData(agent_x, agent_y)

                agent_qps_plots[agent_id].setData(queries_per_second[agent_id])

                # Plot init policy and add it to last list in container.
                if "init policy" in data.keys():
                    init_agent_y = data["init policy"]
                    agents_init_dist[agent_id].setData(agent_x, init_agent_y)

                # Agent q/s.
                agent_qps_x = min(max_x, max(min_x, scaled_bids[agent_id]))
                agents_scatter_qps[agent_id].setData(
                    [agent_qps_x], [queries_per_second[agent_id][-1]]
                )

                # Total queries served by agent
                total_agent_queries_plots[agent_id].setData(
                    total_agent_queries_data[agent_id]
                )

            # Total queries unserved
            total_unserved_queries_plot.setData(total_unserved_queries_data)

            plot_1.setTitle(f"time {i}")

        # 5. Make a step.
        environment.step()
        QtWidgets.QApplication.processEvents()  # type: ignore

        if args.save:
            if i % args.fast_forward_factor == 0:
                if not ffmpeg_process:
                    # Start ffmpeg to save video
                    FILENAME = f"{args.config}.mp4"
                    ffmpeg_process = (
                        ffmpeg.input(
                            "pipe:", format="rawvideo", pix_fmt="rgb24", s="1000x800"
                        )
                        .output(FILENAME, vcodec="libx264", pix_fmt="yuv420p")
                        .overwrite_output()
                        .run_async(pipe_stdin=True)
                    )

                qimage = win.grab().toImage()
                qimage = qimage.convertToFormat(
                    QtGui.QImage.Format_RGB888, QtCore.Qt.AutoColor  # type: ignore
                )

                ffmpeg_process.stdin.write(qimage.constBits().tobytes())

        else:  # Show
            if win.isHidden():
                break

    if ffmpeg_process:
        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()

    if win.isHidden():
        pg.exit()
    else:
        # Keep window open
        pg.exec()


if __name__ == "__main__":
    main()
