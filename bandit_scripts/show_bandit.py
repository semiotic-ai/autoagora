# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from asyncio import run

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from price_multiplier_bandit.price_bandit import (
    ProximalPolicyOptimizationBandit,
    RollingMemContinuousBandit,
    VanillaPolicyGradientBandit,
)
from price_multiplier_bandit.simulated_subgraph import (
    NoisyCyclicQueriesSubgraph,
    NoisyQueriesSubgraph,
)

ITERATIONS = 3500
FAST_FORWARD_FACTOR = 20
BATCH_SIZE = 10

if __name__ == "__main__":
    # Instantiate environment.
    environment = NoisyCyclicQueriesSubgraph()

    # Create bandit.
    bandit = RollingMemContinuousBandit(
        learning_rate=0.01,
        buffer_max_size=BATCH_SIZE,
    )
    FILENAME = f"{bandit}_{environment}.mp4"

    fig, ax = plt.subplots()
    container = []

    # Environment x.
    min_x = 1e-10
    max_x = 3e-6
    env_x = np.linspace(min_x, max_x, 100)

    # Agent x.
    agent_x = np.linspace(-1.4, 1.3, 100)
    agent_x_scaled = [bandit.scale(x) for x in agent_x]

    print("Generating data (with agent trained online)...")
    for i in range(ITERATIONS):
        # X. Collect the values for visualization of non-stationary environment: subgraph queries/price plot.
        if i % FAST_FORWARD_FACTOR == 0:
            y = []
            for val in env_x:
                # Set cost multiplier.
                run(environment.set_cost_multiplier(val))
                # Get observations, i.e. queries per second.
                y.append(run(environment.queries_per_second()))
            (im_env,) = plt.plot(env_x, y, color="r")

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
        if i % FAST_FORWARD_FACTOR == 0:
            policy_mean = bandit.mean.detach().numpy()
            policy_stddev = bandit.logstddev.exp().detach().numpy()
            (img_agent,) = plt.plot(
                agent_x_scaled,
                stats.norm.pdf(agent_x, policy_mean, policy_stddev) * policy_stddev,
                color="b",
            )

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
            legend = ax.legend([im_env, img_agent], ["Queries/s", "Policy PDF"])  # type: ignore
            container.append([im_env, img_agent, title])  # type: ignore

        # 5. Make a step.
        environment.step()

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
