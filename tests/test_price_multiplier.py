import asyncio
from datetime import datetime, timedelta, timezone
from unittest import mock
from unittest.mock import patch

import asyncpg
import pytest
import torch

import autoagora.price_multiplier as price_multiplier
from autoagora.config import args, init_config
from autoagora.price_multiplier import (
    AgentFactory,
    PriceSaveStateDB,
    SubgraphWrapper,
    price_bandit_loop,
    restore_from_save_state,
)
from autoagora.query_metrics import StaticMetricsEndpoints


class TestPriceMultiplier:
    @pytest.fixture
    async def pgpool(self, postgresql):
        pool = await asyncpg.create_pool(
            host=postgresql.info.host,
            database=postgresql.info.dbname,
            user=postgresql.info.user,
            password=postgresql.info.password,
            port=postgresql.info.port,
        )
        assert pool
        await pool.execute(
            """
            CREATE TABLE IF NOT EXISTS price_save_state (
                subgraph        char(46)            PRIMARY KEY,
                last_update     timestamptz         NOT NULL,
                mean            double precision    NOT NULL,
                stddev          double precision    NOT NULL
            )
            """
        )
        yield pool

    async def test_price_bandit_loop(self, pgpool):
        subgraph = "QmTJBvvpknMow6n4YU8R9Swna6N8mHK8N2WufetysBiyuL"
        init_config(
            [
                "--indexer-agent-mgmt-endpoint",
                "http://nowhere",
                "--postgres-host",
                "nowhere",
                "--postgres-username",
                "nowhere",
                "--postgres-password",
                "nowhere",
                "--indexer-service-metrics-endpoint",
                "http://indexer-service.default.svc.cluster.local:7300/metrics",
            ]
        )
        metrics_endpoints = StaticMetricsEndpoints(
            args.indexer_service_metrics_endpoint
        )
        # Mock AgentFactory object and its methods
        with mock.patch("autoagora.price_multiplier.AgentFactory") as mock_AgentFactory:
            # Mock subgraph wrapper methods
            with mock.patch(
                "autoagora.price_multiplier.SubgraphWrapper.set_cost_multiplier"
            ) as mock_set_cost_m:
                mock_set_cost_m.return_value = None
                with mock.patch(
                    "autoagora.price_multiplier.SubgraphWrapper.queries_per_second"
                ) as mock_qps:
                    mock_qps.return_value = 10

                    # Create a task and run it for x secs
                    task = asyncio.create_task(
                        price_bandit_loop(subgraph, pgpool, metrics_endpoints)
                    )
                    # Allowed times a certain function will be called
                    allowed_fn_calls = 50
                    await count_calls(mock_qps, allowed_fn_calls)
                    task.cancel()
                    await pgpool.close()
                    # Assert values for the gauges exist
                    reward_gauge_value = obtain_gauge_value(
                        list(price_multiplier.reward_gauge.collect()), subgraph
                    )
                    price_multiplier_gauge_value = obtain_gauge_value(
                        list(price_multiplier.price_multiplier_gauge.collect()),
                        subgraph,
                    )
                    mean_gauge_value = obtain_gauge_value(
                        list(price_multiplier.mean_gauge.collect()), subgraph
                    )
                    stddev_gauge_value = obtain_gauge_value(
                        list(price_multiplier.stddev_gauge.collect()), subgraph
                    )
                    assert reward_gauge_value
                    assert price_multiplier_gauge_value
                    assert stddev_gauge_value
                    assert mean_gauge_value
                    assert mock_qps.call_count == allowed_fn_calls

                    mock_set_cost_m.assert_called()
                    mock_qps.assert_called()

    async def test_restore_from_save_state(self, pgpool):
        subgraph = "QmTJBvvpknMow6n4YU8R9Swna6N8mHK8N2WufetysBiyuL"
        mean = 0.5
        stddev = 0.3
        time_delta = timedelta(days=3)
        save_state = PriceSaveStateDB(pgpool)
        now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        await pgpool.execute(
            """
            INSERT INTO price_save_state (subgraph, last_update, mean, stddev)
            VALUES ('QmTJBvvpknMow6n4YU8R9Swna6N8mHK8N2WufetysBiyuL', '2023-05-18T21:47:41+00:00', 0.3, 0.2)
            """
        )
        time_to_compare = datetime.strptime(
            "2023-05-19T21:47:41+GMT", "%Y-%m-%dT%H:%M:%S+%Z"
        ).replace(tzinfo=timezone.utc)
        with mock.patch("autoagora.price_multiplier.datetime") as datetime_mock:
            datetime_mock.now.return_value = time_to_compare
            save_state_restore = await restore_from_save_state(
                subgraph, mean, stddev, time_delta, save_state
            )
            assert save_state_restore == (0.3, 0.2)

    async def test_restore_from_save_state_default(self, pgpool):
        subgraph = "QmTJBvvpknMow6n4YU8R9Swna6N8mHK8N2WufetysBiyuL"
        mean = 0.5
        stddev = 0.5
        time_delta = timedelta(days=3)
        save_state = PriceSaveStateDB(pgpool)
        save_state_restore = await restore_from_save_state(
            subgraph, mean, stddev, time_delta, save_state
        )
        assert save_state_restore == (0.5, 0.5)


def obtain_gauge_value(metric_data, subgraph):
    for metric in metric_data:
        for sample in metric.samples:
            if sample.labels["subgraph"] == subgraph:
                value = sample.value
                return value


async def count_calls(mock_obj, count):
    call_count = 0
    while call_count < count:
        await asyncio.sleep(0)
        call_count = mock_obj.call_count
