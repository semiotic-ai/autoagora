import asyncpg
import pytest

from autoagora.logs_db import LogsDB


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
            INSERT INTO query_skeletons (hash, query)
            VALUES ('hash1', 'query getData{ values { id } }'), ('hash2', 'query getInfo{ info { id text} }')
        """
        )
        await pool.execute(
            """
            INSERT INTO query_logs (query_hash, subgraph, timestamp, query_time_ms)
            VALUES ('hash1', 'QmPnu3R7Fm4RmBF21aCYUohDmWbKd3VMXo64ACiRtwUQrn', '2023-05-18T21:47:41+00:00', 100),
            ('hash1', 'QmPnu3R7Fm4RmBF21aCYUohDmWbKd3VMXo64ACiRtwUQrn', '2023-05-18T21:47:41+00:00', 200),
            ('hash2', 'QmTJBvvpknMow6n4YU8R9Swna6N8mHK8N2WufetysBiyuL', '2023-05-18T21:47:41+00:00', 50),
            ('hash1', 'QmTJBvvpknMow6n4YU8R9Swna6N8mHK8N2WufetysBiyuL', '2023-05-18T21:47:41+00:00', 10)
        """
        )
        yield pool
        await pool.close()

    async def test_get_most_frequent_queries_success(self, pgpool):
        ldb = LogsDB(pgpool)
        mfq = await ldb.get_most_frequent_queries(
            "QmPnu3R7Fm4RmBF21aCYUohDmWbKd3VMXo64ACiRtwUQrn", 2
        )
        assert mfq
        compare = LogsDB.QueryStats(
            query="query getData{ values { id } }",
            count=2,
            min_time=100,
            max_time=200,
            avg_time=150,
            stddev_time=70.71067811865476,
        )
        assert mfq[0].query == compare.query
        assert mfq[0].count == compare.count
        assert mfq[0].min_time == compare.min_time
        assert mfq[0].max_time == compare.max_time
        assert mfq[0].avg_time == compare.avg_time
        assert mfq[0].stddev_time == compare.stddev_time

    async def test_get_most_frequent_queries_failed(self, pgpool):
        ldb = LogsDB(pgpool)
        mfq = await ldb.get_most_frequent_queries(
            "QmTJBvvpknMow6n4YU8R9Swna6N8mHK8N2WufetysBiyuL"
        )
        # empty array will be returned since min is default to 100
        assert mfq == []
