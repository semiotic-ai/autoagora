# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from anyio import run

from price_multiplier_bandit.simulated_subgraph import (
    NoisyCyclicQueriesSubgraph,
    NoisyQueriesSubgraph,
)

ITERATIONS = 3000
FAST_FORWARD_FACTOR = 50


async def main():
    # Instantiate environment.
    environment = NoisyCyclicQueriesSubgraph(noise=True)
    FILENAME = f"{environment}.mp4"

    fig, ax = plt.subplots()
    container = []

    # X axis.
    min_x = 1e-10
    max_x = 1e-5
    x = np.linspace(min_x, max_x, 100)

    print("Generating data...")
    for i in range(ITERATIONS // FAST_FORWARD_FACTOR):
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
            f"time {i*FAST_FORWARD_FACTOR}",
            size=plt.rcParams["axes.titlesize"],
            ha="center",
            transform=ax.transAxes,
        )
        container.append([im, title])

        # Make X steps.
        environment.step(number_of_steps=FAST_FORWARD_FACTOR)

    print("Plotting animation...")
    ani = animation.ArtistAnimation(
        fig, container, interval=50, blit=False, repeat_delay=1000
    )

    # Set up formatting for the movie files.
    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=15, metadata=dict(artist="Me"), bitrate=1800)
    plt.show()

    print(f"Saving movie, please wait...")
    ani.save(FILENAME, writer=writer)
    print(f"Movie saved to '{FILENAME}'")


if __name__ == "__main__":
    run(main)
