import asyncio as aio
import json
import random
import re
from datetime import datetime
from unittest import mock

import asyncpg
import pytest

from autoagora.config import init_config
from autoagora.logs_db import LogsDB
from autoagora.model_builder import measure_query_time, mrq_model_builder
from autoagora.utils.constants import MU, SIGMA


async def query_graph_node_mock_side(*args, **kwargs):
    await aio.sleep(random.lognormvariate(MU, SIGMA))
    return mock.MagicMock()


class TestLogsDB:
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
            CREATE TABLE query_skeletons (
                hash BYTEA PRIMARY KEY,
                query TEXT NOT NULL
            )
        """
        )
        await pool.execute(
            """
            CREATE TABLE query_logs (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                query_hash BYTEA REFERENCES query_skeletons(hash),
                subgraph CHAR(46) NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                query_time_ms INTEGER,
                query_variables TEXT

            )
        """
        )
        await pool.execute(
            """
            CREATE TABLE IF NOT EXISTS mrq_query_logs (
                id              uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
                subgraph        char(46)    NOT NULL,
                query_hash      bytea       REFERENCES query_skeletons(hash),
                timestamp       timestamptz NOT NULL,
                query_time_ms   integer,
                query_variables text
            )
            """
        )
        await pool.execute(
            """
            INSERT INTO query_skeletons (hash, query)
            VALUES ('hash1', 'query( $_0: string ){ info( id: $_1 ){ stat val }}'),
            ('hash2', 'query( $_0: string ){ info( id: $_1 ){ stat val }}')
        """
        )
        await pool.execute(
            """
            INSERT INTO query_logs (query_hash, subgraph, timestamp, query_variables)
            VALUES ('hash1', 'QmPnu3R7Fm4RmBF21aCYUohDmWbKd3VMXo64ACiRtwUQrn', '2023-05-18T21:47:41+00:00', $1),
            ('hash1', 'QmPnu3R7Fm4RmBF21aCYUohDmWbKd3VMXo64ACiRtwUQrn', '2023-05-18T21:47:41+00:00', $2),
            ('hash2', 'QmTJBvvpknMow6n4YU8R9Swna6N8mHK8N2WufetysBiyuL', '2023-05-18T21:47:41+00:00', $3),
            ('hash1', 'QmTJBvvpknMow6n4YU8R9Swna6N8mHK8N2WufetysBiyuL', '2023-05-18T21:47:41+00:00', $4)
            """,
            json.dumps(["string_to_insert1", "mock_id1"]),
            json.dumps(["string_to_insert2", "mock_id2"]),
            json.dumps(["string_to_insert3", "mock_id3"]),
            json.dumps(["string_to_insert4", "mock_id4"]),
        )
        yield pool
        await pool.close()

    async def test_get_most_frequent_queries_success(self, pgpool):
        ldb = LogsDB(pgpool)
        mfq = await ldb.get_most_frequent_queries_null_time(
            "QmPnu3R7Fm4RmBF21aCYUohDmWbKd3VMXo64ACiRtwUQrn", 2
        )
        assert mfq
        query = "query {\n  info(id: $_1) {\n    stat\n    val\n  }\n}"
        compare = LogsDB.MRQ_Info(
            hash=b"hash1",
            query=query,
            timestamp=datetime(2023, 9, 19, 10, 59, 10, 463629),
        )
        assert mfq[0].query == compare.query
        assert mfq[0].hash == compare.hash

    async def test_get_measure_query_time(self, pgpool):
        ldb = LogsDB(pgpool)
        mfq = await ldb.get_most_frequent_queries_null_time(
            "QmPnu3R7Fm4RmBF21aCYUohDmWbKd3VMXo64ACiRtwUQrn", 2
        )
        assert mfq
        with mock.patch(
            "autoagora.model_builder.query_graph_node",
            side_effect=query_graph_node_mock_side,
        ) as query_graph_node_mock:
            query_graph_node_mock.return_value = None
            await measure_query_time(
                "QmPnu3R7Fm4RmBF21aCYUohDmWbKd3VMXo64ACiRtwUQrn", mfq[0], ldb, 10
            )
            response = await pgpool.fetch(
                f"""
                SELECT subgraph, query_hash, timestamp, query_time_ms, query_variables 
                FROM mrq_query_logs WHERE subgraph = 'QmPnu3R7Fm4RmBF21aCYUohDmWbKd3VMXo64ACiRtwUQrn'
                """
            )
            # Checking 10 entries where made to the db
            assert len(response) == 10

    async def test_build_mrq_model(self, pgpool, postgresql):
        # This will test almost all of the process, frequent queries is skipped since it takes too much if there are too many values
        with mock.patch(
            "autoagora.logs_db.LogsDB.get_most_frequent_queries_null_time"
        ) as get_mfq_null:
            query = "query {\n  info(id: $_1) {\n    stat\n    val\n  }\n}" "]"
            mfq = [
                LogsDB.MRQ_Info(
                    hash=b"hash1",
                    query=query,
                    timestamp=datetime(2023, 9, 19, 10, 59, 10, 463629),
                )
            ]
            get_mfq_null.return_value = mfq
            with mock.patch(
                "autoagora.model_builder.query_graph_node",
                side_effect=query_graph_node_mock_side,
            ) as query_graph_node_mock:
                query_graph_node_mock.return_value = None
                init_config(
                    [
                        "--indexer-agent-mgmt-endpoint",
                        "http://nowhere",
                        "--postgres-host",
                        postgresql.info.host,
                        "--postgres-username",
                        postgresql.info.user,
                        "--postgres-password",
                        postgresql.info.password,
                        "--postgres-port",
                        str(postgresql.info.port),
                        "--postgres-database",
                        postgresql.info.dbname,
                        "--indexer-service-metrics-endpoint",
                        "http://indexer-service.default.svc.cluster.local:7300/metrics",
                    ]
                )
                model = await mrq_model_builder(
                    "QmPnu3R7Fm4RmBF21aCYUohDmWbKd3VMXo64ACiRtwUQrn", pgpool
                )
                pattern = r"# Generated by AutoAgora \d+\.\d+\.\d+"
                # To ensure a version is being obtained
                assert re.match(
                    pattern, model
                ), f"{model} does not match pattern {pattern}"
                mrq_data = await pgpool.fetch(
                    """
                        SELECT
                            *
                        FROM 
                            mrq_query_logs
                        WHERE
                            subgraph = 'QmPnu3R7Fm4RmBF21aCYUohDmWbKd3VMXo64ACiRtwUQrn'
                        """
                )
                # Should be 100 since is the amount of iterations made
                assert len(mrq_data) == 100

    async def test_get_most_frequent_queries_failed(self, pgpool):
        ldb = LogsDB(pgpool)
        mfq = await ldb.get_most_frequent_queries(
            "QmTJBvvpknMow6n4YU8R9Swna6N8mHK8N2WufetysBiyuL"
        )
        # empty array will be returned since min is default to 100
        assert mfq == []
