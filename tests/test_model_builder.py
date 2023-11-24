import asyncio
import json
import os
import random
import re
import tempfile
from datetime import datetime
from typing import Mapping, Optional
from unittest import mock

import psycopg_pool
import pytest
from psycopg import sql

from autoagora.config import init_config
from autoagora.logs_db import LogsDB
from autoagora.model_builder import (
    apply_default_model,
    build_template,
    mrq_model_builder,
)
from tests.utils.constants import TEST_MANUAL_AGORA_ENTRY, TEST_QUERY_1, TEST_QUERY_2


async def wait_random_time(query: str, variables: Optional[Mapping] = None):
    await asyncio.sleep(random.lognormvariate(-4, 0.3))


class TestModelBuilder:
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
        async with pool.connection() as conn:
            await conn.execute(
                """
                CREATE TABLE query_skeletons (
                    hash BYTEA PRIMARY KEY,
                    query TEXT NOT NULL
                )
            """
            )
            await conn.execute(
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
            await conn.execute(
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
            await conn.execute(
                """
                INSERT INTO query_skeletons (hash, query)
                VALUES ('hash1', 'query( $_0: string ){ info( id: $_1 ){ stat val }}'),
                ('hash2', 'query( $_0: string ){ info( id: $_1 ){ stat val }}')
            """
            )
            await conn.execute(
                sql.SQL(
                    """
                INSERT INTO query_logs (query_hash, subgraph, timestamp, query_variables)
                VALUES ('hash1', 'QmPnu3R7Fm4RmBF21aCYUohDmWbKd3VMXo64ACiRtwUQrn', '2023-05-18T21:47:41+00:00', {mock1}),
                ('hash1', 'QmPnu3R7Fm4RmBF21aCYUohDmWbKd3VMXo64ACiRtwUQrn', '2023-05-18T21:47:41+00:00', {mock2}),
                ('hash2', 'QmTJBvvpknMow6n4YU8R9Swna6N8mHK8N2WufetysBiyuL', '2023-05-18T21:47:41+00:00', {mock3}),
                ('hash1', 'QmTJBvvpknMow6n4YU8R9Swna6N8mHK8N2WufetysBiyuL', '2023-05-18T21:47:41+00:00', {mock4})
                """
                ).format(
                    mock1=json.dumps(["string_to_insert1", "mock_id1"]),
                    mock2=json.dumps(["string_to_insert2", "mock_id2"]),
                    mock3=json.dumps(["string_to_insert3", "mock_id3"]),
                    mock4=json.dumps(["string_to_insert4", "mock_id4"]),
                )
            )
        yield pool
        await pool.close()

    async def test_build_model(self, postgresql):
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
        most_frequent_queries = [
            LogsDB.QueryStats(
                query=TEST_QUERY_1,
                count=100,
                min_time=1,
                max_time=60,
                avg_time=1.2,
                stddev_time=0.5,
            ),
            LogsDB.QueryStats(
                query=TEST_QUERY_2,
                count=10,
                min_time=3,
                max_time=20,
                avg_time=13.2,
                stddev_time=0.7,
            ),
        ]

        model = build_template(
            "Qmadj8x9km1YEyKmRnJ6EkC2zpJZFCfTyTZpuqC3j6e1QH", most_frequent_queries
        )
        pattern = r"# Generated by AutoAgora \d+\.\d+\.\d+"
        # To ensure a version is being obtained
        assert re.match(pattern, model), f"{model} does not match pattern {pattern}"
        assert TEST_QUERY_1 in model
        assert TEST_QUERY_2 in model
        assert TEST_MANUAL_AGORA_ENTRY not in model

    async def test_build_model_with_manual_entry(self, postgresql):
        subgraph = "Qmadj8x9km1YEyKmRnJ6EkC2zpJZFCfTyTZpuqC3j6e1QH"
        file_type = ".agora"
        # Creates a temp dir to simulate manual entries
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a file inside temp dir which will be deleted once out of context
            with open(os.path.join(temp_dir, subgraph + file_type), "w") as temp_file:
                temp_file.write(TEST_MANUAL_AGORA_ENTRY)
            most_frequent_queries = [
                LogsDB.QueryStats(
                    query=TEST_QUERY_1,
                    count=100,
                    min_time=1,
                    max_time=60,
                    avg_time=1.2,
                    stddev_time=0.5,
                ),
                LogsDB.QueryStats(
                    query=TEST_QUERY_2,
                    count=10,
                    min_time=3,
                    max_time=20,
                    avg_time=13.2,
                    stddev_time=0.7,
                ),
            ]
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
                    "--manual-entry-path",
                    temp_dir,
                ]
            )
            model = build_template(subgraph, most_frequent_queries)
            pattern = r"# Generated by AutoAgora \d+\.\d+\.\d+"
            # To ensure a version is being obtained
            assert re.match(pattern, model), f"{model} does not match pattern {pattern}"
            assert TEST_QUERY_1 in model
            assert TEST_QUERY_2 in model
            assert TEST_MANUAL_AGORA_ENTRY in model

    async def test_build_mrq_model(self, pgpool, postgresql):
        with mock.patch(
            "autoagora.logs_db.LogsDB.get_most_frequent_queries_null_time"
        ) as mock_get__mrq_mfq:
            with mock.patch(
                "autoagora.model_builder.query_graph_node"
            ) as mock_query_graph_node:
                mock_get__mrq_mfq.return_value = [
                    LogsDB.MRQ_Info(
                        hash=b"hash1",
                        query="query( $_0: string ){ info( id: $_1 ){ stat val }}",
                        timestamp=datetime.now(),
                    )
                ]
                mock_query_graph_node.side_effect = wait_random_time
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
                # assert model == 1
                async with pgpool.connection() as connection:
                    mrq_data = await connection.execute(
                        sql.SQL(
                            """
                            SELECT
                                *
                            FROM 
                                mrq_query_logs
                            """
                        )
                    )
                    mrq_data = await mrq_data.fetchall()
                    assert mrq_data

    async def test_apply_default_model(self, postgresql):
        subgraph = "Qmadj8x9km1YEyKmRnJ6EkC2zpJZFCfTyTZpuqC3j6e1QH"
        file_type = ".agora"
        # Creates a temp dir to simulate manual entries
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a file inside temp dir which will be deleted once out of context
            with open(os.path.join(temp_dir, subgraph + file_type), "w") as temp_file:
                temp_file.write(TEST_MANUAL_AGORA_ENTRY)
            with mock.patch(
                "autoagora.model_builder.set_cost_model"
            ) as set_cost_model_mock:
                set_cost_model_mock.return_value = None
                with mock.patch("logging.debug") as logging_debug_mock:
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
                            "--manual-entry-path",
                            temp_dir,
                        ]
                    )
                    await apply_default_model(subgraph)
                    # Obtain the args send to the logger.debug fn
                    debug_logs_args = logging_debug_mock.call_args[0]
                    # Remove \n since they cause the assertion to bug and fail
                    debug_logs_args = str(debug_logs_args).replace(r"\n", "")
                    manual_agora_entry = TEST_MANUAL_AGORA_ENTRY.replace("\n", "")
                    assert manual_agora_entry in debug_logs_args
