# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Tuple

import asyncpg
from autoagora_agents.agent_factory import AgentFactory
from prometheus_client import Gauge

from autoagora.config import args
from autoagora.price_save_state_db import PriceSaveStateDB
from autoagora.query_metrics import MetricsEndpoints
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


async def price_bandit_loop(
    subgraph: str, pgpool: asyncpg.Pool, metrics_endpoints: MetricsEndpoints
):
    try:
        # Instantiate environment.
        environment = SubgraphWrapper(subgraph)

        # Try restoring the mean and stddev from a save state, or use defaults
        save_state_db = PriceSaveStateDB(pgpool)
        start_mean, start_stddev = await restore_from_save_state(
            subgraph=subgraph,
            default_mean=5e-8,
            default_stddev=1e-1,
            max_save_state_age=timedelta(hours=24),
            save_state_db=save_state_db,
        )

        agent_section = {
            "policy": {"type": "rolling_ppo", "buffer_max_size": 10},
            "action": {
                "type": "scaled_gaussian",
                "initial_mean": start_mean,
                "initial_stddev": start_stddev,
            },
            "optimizer": {"type": "adam", "lr": 0.01},
        }

        bandit = AgentFactory(
            agent_name="RollingMemContinuousBandit", agent_section=agent_section
        )

        total_revenue = 0

        print("Training agent. Please wait...")

        while True:
            logging.debug(
                "Price bandit %s - Distribution mean: %s",
                subgraph,
                bandit.mean().item(),
            )
            mean_gauge.labels(subgraph=subgraph).set(
                bandit.bid_scale(bandit.mean().item())
            )
            logging.debug(
                "Price bandit %s - Distribution stddev: %s",
                subgraph,
                bandit.stddev().item(),
            )
            stddev_gauge.labels(subgraph=subgraph).set(bandit.stddev().item())

            # Update the save state
            # NOTE: `bid_scale` is specific to "scaled_gaussian" agent action type
            logging.debug("Price bandit %s - Saving state to DB.", subgraph)
            await save_state_db.save_state(
                subgraph=subgraph,
                mean=bandit.bid_scale(bandit.mean().item()),
                stddev=bandit.stddev().item(),
            )

            # 1. Get bid from the agent (action)
            scaled_bid = bandit.get_action()

            logging.debug(
                "Price bandit %s - Price multiplier: %s", subgraph, scaled_bid
            )
            price_multiplier_gauge.labels(subgraph=subgraph).set(scaled_bid)

            # 2. Act: set multiplier in the environment.
            await environment.set_cost_multiplier(scaled_bid)

            # 3. Get the reward.
            # Get queries per second.
            queries_per_second = await environment.queries_per_second(
                metrics_endpoints, args.qps_observation_duration
            )
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
            logging.debug(
                "Price bandit %s - Total revenue: %s", subgraph, total_revenue
            )

            # Add reward.
            bandit.add_reward(revenue_per_second)

            # 4. Update the policy.
            loss = bandit.update_policy()
            if loss is not None:
                logging.debug("Price bandit %s - Training loss: %s", subgraph, loss)

    except asyncio.CancelledError as cancelledError:
        logging.debug("Price bandit %s - Removing bandit loop", subgraph)
        raise cancelledError
    except:
        logging.exception("price_bandit_loop error")
        exit(-1)


async def restore_from_save_state(
    subgraph: str,
    default_mean: float,
    default_stddev: float,
    max_save_state_age: timedelta,
    save_state_db: PriceSaveStateDB,
) -> Tuple[float, float]:
    """Restore a subgraph's price mean and stddev from the save state database.

    If save_state_db is None **OR** there is no save state for the subgraph **OR** the
    found save state is older than max_save_state_age, return the given default values.

    Args:
        subgraph (str): Subgraph IPFS hash.
        default_mean (float): Default price mean if no eligible save state.
        default_stddev (float): Default price stddev if no eligible save state.
        max_save_state_age (timedelta): Maximum age of the save state.
        save_state_db (Optional[PriceSaveStateDB]): Save state database wrapper.

    Returns:
        Tuple[float, float]: Price mean and stddev.
    """

    mean = default_mean
    stddev = default_stddev

    if save_state_db:
        save_state = await save_state_db.load_state(subgraph)
        # If there is a save state for that subgraph
        if save_state:
            # If the save state is not older than max_save_state_age
            if datetime.now(timezone.utc) - save_state.last_update < max_save_state_age:
                mean = save_state.mean
                stddev = save_state.stddev

    return mean, stddev
