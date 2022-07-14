# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
from pydoc import describe

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from anyio import run

from price_multiplier_bandit.environment_factory import (
    EnvironmentFactory,
    add_environment_argparse,
)


def add_experiment_argparse(parser: argparse):
    """Adds argparse arguments related to experiment to parser."""
    parser.add_argument(
        "-i",
        "--iterations",
        default=3000,
        type=int,
        help="Sets the length of the experiment / number of args.iterations (DEFAULT: 3000)",
    )
    parser.add_argument(
        "-f",
        "--fast-forward-factor",
        default=50,
        type=int,
        help="Sets the fast forward factor (DEFAULT: 50)",
    )
    parser.add_argument(
        "--show", action="store_true", help="If set, shows the animation"
    )
    parser.add_argument(
        "--save", action="store_true", help="If set, saves the animation to a file"
    )


async def main():
    # Init argparse.
    parser = argparse.ArgumentParser(
        usage="%(prog)s [-e ...] [-n ...] [--show] [--save]",
        description="Runs subgraph simulation and (optionally) shows it and/or saves it to a file.",
    )
    add_experiment_argparse(parser=parser)
    add_environment_argparse(parser=parser)

    # Parse arguments
    args = parser.parse_args()

    # Instantiate environment.
    environment = EnvironmentFactory(environment_type=args.environment)
    FILENAME = f"{environment}.mp4"

    fig, ax = plt.subplots()
    container = []

    # X axis.
    min_x = 1e-10
    max_x = 1e-5
    x = np.linspace(min_x, max_x, 100)

    print(f"Generating {environment}. Please wait...")
    for i in range(args.iterations // args.fast_forward_factor):
        y = []
        for val in x:
            # Set cost multiplier.
            await environment.set_cost_multiplier(val)
            # Get observations, i.e. queries per second.
            y.append(await environment.queries_per_second())

        # Plot both.
        (im,) = plt.plot(x, y, color="r")

        ax.set_xlabel("price multiplier")
        ax.set_ylabel("queries/second")
        title = ax.text(
            0.5,
            1.05,
            f"time {i*args.fast_forward_factor}",
            size=plt.rcParams["axes.titlesize"],
            ha="center",
            transform=ax.transAxes,
        )
        container.append([im, title])

        # Make X steps.
        environment.step(number_of_steps=args.fast_forward_factor)

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


if __name__ == "__main__":
    run(main)
