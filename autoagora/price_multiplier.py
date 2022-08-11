# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import logging

from prometheus_client import Gauge

from agents.reinforcement_learning_bandit import RollingMemContinuousBandit
from autoagora.subgraph_wrapper import SubgraphWrapper

reward_gauge = Gauge(
    "bandit_reward",
    "Reward of the bandit training: queries_per_second * price_multiplier.",
    ["subgraph"],
)
price_multiplier_gauge = Gauge(
    "bandit_price_multiplier",
    "Price multiplier sampled from the Gaussian model.",
    ["subgraph"],
)
stddev_gauge = Gauge(
    "bandit_stddev",
    "Standard deviation of the Gaussian price multiplier model.",
    ["subgraph"],
)
mean_gauge = Gauge(
    "bandit_mean", "Mean of the Gaussian price multiplier model.", ["subgraph"]
)


async def price_bandit_loop(subgraph: str):
    # Instantiate environment.
    environment = SubgraphWrapper(subgraph)

    bandit = RollingMemContinuousBandit(
        learning_rate=0.01,
        initial_mean=-3,
        initial_logstddev=1,
        buffer_max_size=10,
    )

    total_revenue = 0

    print("Training agent. Please wait...")

    while True:
        logging.debug(
            "Price bandit %s - Distribution mean: %s", subgraph, bandit.mean.item()
        )
        mean_gauge.labels(subgraph=subgraph).set(bandit.scale(bandit.mean.item()))
        logging.debug(
            "Price bandit %s - Distribution stddev: %s",
            subgraph,
            bandit.logstddev.exp().item(),
        )
        stddev_gauge.labels(subgraph=subgraph).set(bandit.logstddev.exp().item())

        # 1. Get bid from the agent (action)
        scaled_bid = bandit.get_action()

        logging.debug("Price bandit %s - Price multiplier: %s", subgraph, scaled_bid)
        price_multiplier_gauge.labels(subgraph=subgraph).set(scaled_bid)

        # 2. Act: set multiplier in the environment.
        await environment.set_cost_multiplier(scaled_bid)

        # 3. Get the reward.
        # Get queries per second.
        queries_per_second = await environment.queries_per_second(60)
        logging.debug(
            "Price bandit %s - Queries per second: %s", subgraph, queries_per_second
        )

        # Turn it into revenue. Actually this is just a monotonically increasing
        # function of the actual GRT revenue.
        revenue_per_second = queries_per_second * scaled_bid
        logging.debug(
            "Price bandit %s - Revenue per second: %s", subgraph, revenue_per_second
        )
        reward_gauge.labels(subgraph=subgraph).set(revenue_per_second)
        total_revenue += revenue_per_second
        logging.debug("Price bandit %s - Total revenue: %s", subgraph, total_revenue)

        # Add reward.
        bandit.add_reward(revenue_per_second)

        # 4. Update the policy.
        loss = bandit.update_policy()
        if loss is not None:
            logging.debug("Price bandit %s - Training loss: %s", subgraph, loss)
