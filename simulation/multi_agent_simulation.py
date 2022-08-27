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
    pg.setConfigOption("background", "white")
    pg.setConfigOption("foreground", "black")
    pg.setConfigOptions(antialias=True)
    app = pg.mkQApp("Plot")
    win = pg.GraphicsLayoutWidget(show=not args.save, title="Multi-agent training")
    win.resize(1000, 800)

    # Create policy plot
    plot_1 = win.addPlot(title="time 0")
    plot_1.setPreferredHeight(700)
    plot_1.addLegend(offset=(-1, 1))
    plot_1.setYRange(0, 1.3)
    plot_1.setLabel("left", "Queries/s")
    plot_1.setLabel("bottom", "Price multiplier")
    # Policy PD
    agents_dist = [
        plot_1.plot(
            pen=pg.mkPen(color=(i, len(agents)), width=2),
            name=f"Agent {agent_name}: policy",
        )
        for i, agent_name in enumerate(agents.keys())
    ]
    # Initial policy PD
    agents_init_dist = [
        plot_1.plot(
            pen=pg.mkPen(color=(i, len(agents)), width=2, style=QtCore.Qt.DotLine),  # type: ignore
            name=f"Agent {agent_name}: init policy",
        )
        for i, agent_name in enumerate(agents.keys())
    ]
    # Environment QPS
    env_plot = plot_1.plot(
        pen=pg.mkPen(color="gray", width=2), name="Environment: total q/s"
    )
    agents_scatter_qps = pg.ScatterPlotItem()
    plot_1.addItem(agents_scatter_qps)

    win.nextRow()

    # Create query volume bar graph
    plot_2 = win.addPlot()
    plot_2.setPreferredHeight(300)
    plot_2.setXRange(0, 1.5)
    plot_2.setLabel("left", "Agent #")
    plot_2.setLabel("bottom", "Queries/s")
    y = [i for i in range(len(agents))]
    w = [0 for _ in range(len(agents))]
    qps_bars = pg.BarGraphItem(
        x0=0,
        y=y,
        height=0.7,
        width=w,
        brushes=[(i, len(agents)) for i in range(len(agents))],
    )
    plot_2.addItem(qps_bars)

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
            agents_scatter_qps.clear()

            for agent_id, (agent_name, agent) in enumerate(agents.items()):

                # Get data.
                data = run(agent.generate_plot_data(min_x, max_x))
                agent_x = data.pop("x")
                agent_y = data["policy"]
                agents_dist[agent_id].setData(agent_x, agent_y)

                qps_bars.setOpts(width=queries_per_second)

                # Plot init policy and add it to last list in container.
                if "init policy" in data.keys():
                    init_agent_y = data["init policy"]
                    agents_init_dist[agent_id].setData(agent_x, init_agent_y)

                # Agent q/s.
                agent_qps_x = min(max_x, max(min_x, scaled_bids[agent_id]))
                agents_scatter_qps.addPoints(
                    [
                        {
                            "pos": (agent_qps_x, queries_per_second[agent_id]),
                            "brush": (agent_id, len(agents)),
                            "pen": pg.mkPen(color=(agent_id, len(agents)), width=1),
                        }
                    ]
                )

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

    pg.exit()


if __name__ == "__main__":
    main()
