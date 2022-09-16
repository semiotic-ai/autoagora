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

LOG_PLOT = True
WINDOW_SIZE = (1000, 1000)


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
    win.resize(*WINDOW_SIZE)

    # Create policy plot
    policy_plot = win.addPlot(title="time 0")
    policy_plot.setPreferredHeight(300)
    policy_plot_legend = policy_plot.addLegend(offset=None)
    policy_plot_vb = (
        win.addViewBox()
    )  # Empty UI box to contain the legend outside the plot
    policy_plot_vb.setFixedWidth(300)
    policy_plot_legend.setParentItem(policy_plot_vb)
    policy_plot.setYRange(0, 1.1)
    policy_plot.setXRange(min_x, max_x)
    policy_plot.setClipToView(True)
    policy_plot.setLabel("left", "Query rate")
    policy_plot.setLabel("bottom", "Price multiplier")
    policy_plot.setLogMode(LOG_PLOT, False)
    # Policy PD
    agents_dist = [
        policy_plot.plot(
            pen=pg.mkPen(color=(i, len(agents) + 1), width=1.5),
            name=f"Agent {agent_name}: policy",
        )
        for i, agent_name in enumerate(agents.keys())
    ]

    # Initial policy PD
    agents_init_dist = [
        policy_plot.plot(
            pen=pg.mkPen(color=(i, len(agents) + 1), width=1.5, style=QtCore.Qt.DotLine),  # type: ignore
            name=f"Agent {agent_name}: init policy",
        )
        for i, agent_name in enumerate(agents.keys())
    ]
    # This is a line plot with invisible line and visible data points.
    # Easier to scale with the rest of the plot than with using a ScatterPlot.
    agents_scatter_qps = [
        policy_plot.plot(
            pen=pg.mkPen(color=(0, 0, 0, 0), width=0),  # type: ignore
            name=f"Agent {agent_name}: query rate",
            symbolBrush=(i, len(agents) + 1),
            symbolPen="w",
        )
        for i, agent_name in enumerate(agents.keys())
    ]
    # Environment QPS
    env_plot = policy_plot.plot(
        pen=pg.mkPen(color="gray", width=1.5), name="Environment: total query rate"
    )

    win.nextRow()

    # Create query volume time plot
    query_rate_plot = win.addPlot()
    # query_rate_plot.setPreferredHeight(200)
    query_rate_plot.setLabel("left", "Query rate")
    query_rate_plot.setLabel("bottom", "Timestep")
    query_rate_plot_legend = query_rate_plot.addLegend(offset=None)
    query_rate_plot_vb = win.addViewBox()
    query_rate_plot_vb.setFixedWidth(300)
    query_rate_plot_legend.setParentItem(query_rate_plot_vb)
    agent_qps_plots = [
        query_rate_plot.plot(
            pen=pg.mkPen(color=(i, len(agents) + 1), width=1.5),
            name=f"Agent {agent_name}",
        )
        for i, agent_name in enumerate(agents.keys())
    ]
    queries_per_second = [[] for _ in agents]

    win.nextRow()

    # Create total queries (un)served plot
    total_queries_plot = win.addPlot()
    # total_queries_plot.setPreferredHeight(200)
    total_queries_plot.setLabel("left", "Total queries")
    total_queries_plot.setLabel("bottom", "Timestep")
    total_queries_legend = total_queries_plot.addLegend(offset=None)
    total_queries_vb = win.addViewBox()
    total_queries_vb.setFixedWidth(300)
    total_queries_legend.setParentItem(total_queries_vb)
    total_queries_legend.anchor((0, 0), (0, 0))

    total_agent_queries_plots = [
        total_queries_plot.plot(
            pen=pg.mkPen(color=(i, len(agents) + 1), width=1.5),
            name=f"Agent {agent_name}",
        )
        for i, agent_name in enumerate(agents.keys())
    ]
    total_unserved_queries_plot = total_queries_plot.plot(
        pen=pg.mkPen(color=(len(agents), len(agents) + 1), width=1.5),
        name=f"Dropped",
    )

    total_agent_queries_data = [[] for _ in agents]
    total_unserved_queries_data = []

    win.nextRow()

    # Create revenue rate plot
    revenue_rate_plot = win.addPlot()
    # revenue_rate_plot.setPreferredHeight(200)
    revenue_rate_plot.setLabel("left", "Revenue rate")
    revenue_rate_plot.setLabel("bottom", "Timestep")
    revenue_rate_legend = revenue_rate_plot.addLegend(offset=None)
    revenue_rate_vb = win.addViewBox()
    revenue_rate_vb.setFixedWidth(300)
    revenue_rate_legend.setParentItem(revenue_rate_vb)
    revenue_rate_legend.anchor((0, 0), (0, 0))

    revenue_rate_plots = [
        revenue_rate_plot.plot(
            pen=pg.mkPen(color=(i, len(agents) + 1), width=1.5),
            name=f"Agent {agent_name}",
        )
        for i, agent_name in enumerate(agents.keys())
    ]

    revenue_rate_data = [[] for _ in agents]

    win.nextRow()

    # Create total revenue plot
    total_revenue_plot = win.addPlot()
    # total_revenue_plot.setPreferredHeight(200)
    total_revenue_plot.setLabel("left", "Total revenue")
    total_revenue_plot.setLabel("bottom", "Timestep")
    total_revenue_legend = total_revenue_plot.addLegend(offset=None)
    total_revenue_vb = win.addViewBox()
    total_revenue_vb.setFixedWidth(300)
    total_revenue_legend.setParentItem(total_revenue_vb)
    total_revenue_legend.anchor((0, 0), (0, 0))

    total_revenue_plots = [
        total_revenue_plot.plot(
            pen=pg.mkPen(color=(i, len(agents) + 1), width=1.5),
            name=f"Agent {agent_name}",
        )
        for i, agent_name in enumerate(agents.keys())
    ]

    total_revenue_data = [[] for _ in agents]

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

        # Make a step. (Executes a number of queries in the case of the ISA)
        environment.step()

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

            revenue_rate_data[agent_id] += [monies_per_second]
            if i > 0:
                total_revenue_data[agent_id] += [
                    total_revenue_data[agent_id][-1] + revenue_rate_data[agent_id][-1]
                ]
            else:
                total_revenue_data[agent_id] += [revenue_rate_data[agent_id][-1]]

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
        if i > 0:
            total_unserved_queries_data += [
                total_unserved_queries_data[-1]
                + 1
                - sum(e[-1] for e in queries_per_second)
            ]
        else:
            total_unserved_queries_data += [1 - sum(e[-1] for e in queries_per_second)]

        # X. Collect the values for visualization of agent's gaussian policy.
        if i % args.fast_forward_factor == 0:
            for agent_id, (agent_name, agent) in enumerate(agents.items()):

                # Get data.
                data = run(agent.generate_plot_data(min_x, max_x, logspace=LOG_PLOT))
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

                # Revenue rate by agent
                revenue_rate_plots[agent_id].setData(revenue_rate_data[agent_id])

                # Total revenue by agent
                total_revenue_plots[agent_id].setData(total_revenue_data[agent_id])

            # Total queries unserved
            total_unserved_queries_plot.setData(total_unserved_queries_data)

            policy_plot.setTitle(f"time {i}")

        QtWidgets.QApplication.processEvents()  # type: ignore

        if args.save:
            if i % args.fast_forward_factor == 0:
                if not ffmpeg_process:
                    # Start ffmpeg to save video
                    FILENAME = f"{args.config}.mp4"
                    ffmpeg_process = (
                        ffmpeg.input(
                            "pipe:",
                            format="rawvideo",
                            pix_fmt="rgb24",
                            s=f"{WINDOW_SIZE[0]}x{WINDOW_SIZE[1]}",
                        )
                        .output(FILENAME, vcodec="libx264", pix_fmt="yuv420p")
                        .overwrite_output()
                        .run_async(pipe_stdin=True)
                    )

                qimage = win.grab().toImage()
                qimage = qimage.convertToFormat(
                    QtGui.QImage.Format_RGB888, QtCore.Qt.AutoColor  # type: ignore
                )

                # May have to rescale (HiDPI displays, etc)
                if (qimage.width(), qimage.height()) != WINDOW_SIZE:
                    qimage = (
                        QtGui.QPixmap.fromImage(qimage)  # type: ignore
                        .scaled(
                            WINDOW_SIZE[0],
                            WINDOW_SIZE[1],
                            mode=QtCore.Qt.TransformationMode.SmoothTransformation,  # type: ignore
                        )
                        .toImage()
                        .convertToFormat(
                            QtGui.QImage.Format_RGB888, QtCore.Qt.AutoColor  # type: ignore
                        )
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
