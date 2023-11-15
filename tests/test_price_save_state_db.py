import random
import string

import psycopg_pool
import pytest
from autoagora_agents.agent_factory import AgentFactory
from numpy.testing import assert_approx_equal

from autoagora import price_save_state_db


class TestPriceSaveStateDB:
    @pytest.fixture
    async def pgpool(self, postgresql):
        conn_string = (
            f"host={postgresql.info.host} "
            f"dbname={postgresql.info.dbname} "
            f"user={postgresql.info.user} "
            f'password="{postgresql.info.password}" '
            f"port={postgresql.info.port}"
        )

        pool = psycopg_pool.AsyncConnectionPool(
            conn_string, min_size=2, max_size=10, open=False
        )
        await pool.open()
        await pool.wait()
        yield pool
        await pool.close()

    @pytest.fixture
    async def pssdb(self, pgpool):
        pssdb_ = price_save_state_db.PriceSaveStateDB(pgpool)
        yield pssdb_

    async def test_create_write_read(self, pssdb):
        random.seed(42)

        # Create 30 to 50 random entries
        subgraph_price_list = [
            (
                "".join(random.choices(string.ascii_letters + string.digits, k=46)),
                random.random() * 1e-8,
                random.random(),
            )
            for _ in range(random.randrange(30, 50))
        ]

        # Add the entries to the save state DB
        for subgraph_price in subgraph_price_list:
            await pssdb.save_state(*subgraph_price)

        # Pick 10 random subgraphs, retrieve their save state from DB, compare to
        # ground truth.
        print("Checking 10 subgraph states at random...")
        for subgraph_price_truth in random.choices(subgraph_price_list, k=10):
            subgraph_price_from_db = await pssdb.load_state(subgraph_price_truth[0])
            print(f"{subgraph_price_from_db=}")
            print(f"{subgraph_price_truth=}")
            assert subgraph_price_from_db.mean == subgraph_price_truth[1]
            assert subgraph_price_from_db.stddev == subgraph_price_truth[2]

    async def test_update_entry(self, pssdb):
        random.seed(42)

        # Create 10 random entries
        subgraph_price_list = [
            (
                "".join(random.choices(string.ascii_letters + string.digits, k=46)),
                random.random() * 1e-8,
                random.random(),
            )
            for _ in range(10)
        ]

        # Add the entries to the save state DB
        for subgraph_price in subgraph_price_list:
            await pssdb.save_state(*subgraph_price)

        # Randomly update the entries 20 times
        print("Randomly updating subgraph save states...")
        for index in random.choices(range(len(subgraph_price_list)), k=20):
            subgraph = subgraph_price_list[index]
            print(f"Before {subgraph}")
            subgraph_price_list[index] = (
                subgraph[0],
                random.random() * 1e-8,
                random.random(),
            )
            print(f"After {subgraph_price_list[index]}")

            await pssdb.save_state(*subgraph_price_list[index])

        # Retrieve all the save states from DB, compare to ground truth.
        print("Checking all the subgraph save states...")
        for subgraph_price_truth in subgraph_price_list:
            subgraph_price_from_db = await pssdb.load_state(subgraph_price_truth[0])
            print(f"{subgraph_price_from_db=}")
            print(f"{subgraph_price_truth=}")
            assert subgraph_price_from_db.mean == subgraph_price_truth[1]
            assert subgraph_price_from_db.stddev == subgraph_price_truth[2]

    async def test_save_reload_agent(self, pssdb):
        random.seed(42)

        mean = 5e-8
        stddev = 1e-1

        subgraph = "".join(random.choices(string.ascii_letters + string.digits, k=46))

        # Create pricing agent
        agent_section = {
            "policy": {"type": "rolling_ppo", "buffer_max_size": 10},
            "action": {
                "type": "scaled_gaussian",
                "initial_mean": mean,
                "initial_stddev": stddev,
            },
            "optimizer": {"type": "adam", "lr": 0.01},
        }
        bandit = AgentFactory(
            agent_name="RollingMemContinuousBandit", agent_section=agent_section
        )

        assert_approx_equal(actual=bandit.bid_scale(bandit.mean().item()), desired=mean)
        assert_approx_equal(actual=bandit.stddev().item(), desired=stddev)

        await pssdb.save_state(
            subgraph=subgraph,
            mean=bandit.bid_scale(bandit.mean().item()),
            stddev=bandit.stddev().item(),
        )

        del bandit

        save_state = await pssdb.load_state(subgraph)

        agent_section = {
            "policy": {"type": "rolling_ppo", "buffer_max_size": 10},
            "action": {
                "type": "scaled_gaussian",
                "initial_mean": save_state.mean,
                "initial_stddev": save_state.stddev,
            },
            "optimizer": {"type": "adam", "lr": 0.01},
        }
        bandit = AgentFactory(
            agent_name="RollingMemContinuousBandit", agent_section=agent_section
        )

        assert_approx_equal(actual=bandit.bid_scale(bandit.mean().item()), desired=mean)
        assert_approx_equal(actual=bandit.stddev().item(), desired=stddev)
