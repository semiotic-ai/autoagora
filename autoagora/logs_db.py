# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

import asyncpg
import configargparse

from autoagora.config import args

argsparser = configargparse.get_argument_parser()
argsparser.add_argument(
    "--logs-postgres-host",
    env_var="LOGS_POSTGRES_HOST",
    required=True,
    help="Host of the postgres instance storing the logs.",
)
argsparser.add_argument(
    "--logs-postgres-port",
    env_var="LOGS_POSTGRES_PORT",
    required=False,
    type=int,
    default=5432,
    help="Port of the postgres instance storing the logs.",
)
argsparser.add_argument(
    "--logs-postgres-database",
    env_var="LOGS_POSTGRES_DATABASE",
    required=True,
    help="Name of the logs database.",
)
argsparser.add_argument(
    "--logs-postgres-username",
    env_var="LOGS_POSTGRES_USERNAME",
    required=True,
    help="Username for the logs database.",
)
argsparser.add_argument(
    "--logs-postgres-password",
    env_var="LOGS_POSTGRES_PASSWORD",
    required=True,
    help="Password for the logs database.",
)


class LogsDB:
    @dataclass
    class QueryStats:
        query: str
        count: int
        min_time: int
        max_time: int
        avg_time: float
        stddev_time: float

    def __init__(self) -> None:
        self.connection: Optional[asyncpg.connection.Connection] = None

    async def connect(self) -> None:
        self.connection = await asyncpg.connect(
            host=args.logs_postgres_host,
            database=args.logs_postgres_database,
            user=args.logs_postgres_username,
            password=args.logs_postgres_password,
            port=args.logs_postgres_port,
        )

    async def get_most_frequent_queries(
        self, subgraph_ipfs_hash: str, min_count: int = 100
    ):
        assert self.connection
        rows = await self.connection.fetch(
            """
            SELECT
                query,
                count_id,
                min_time,
                max_time,
                avg_time,
                stddev_time
            FROM
                query_skeletons
            INNER JOIN
            (
                SELECT
                    query_hash as qhash,
                    count(id) as count_id,
                    Min(query_time_ms) as min_time,
                    Max(query_time_ms) as max_time,
                    Avg(query_time_ms) as avg_time,
                    Stddev(query_time_ms) as stddev_time 
                FROM
                    query_logs
                WHERE
                    subgraph = $1
                    AND query_time_ms IS NOT NULL
                GROUP BY
                    qhash
                HAVING
                    Count(id) >= $2
            ) as query_logs
            ON
                qhash = hash
            ORDER BY
                count_id DESC
            """,
            subgraph_ipfs_hash,
            min_count,
        )

        return [
            LogsDB.QueryStats(
                query=row[0],
                count=row[1],
                min_time=row[2],
                max_time=row[3],
                avg_time=float(row[4]),
                stddev_time=float(row[5]),
            )
            for row in rows
        ]

    async def get_subgraph_average_query_stats(self, subgraph_ipfs_hash: str):
        assert self.connection
        row = await self.connection.fetchrow(
            """
            SELECT
                count(id),
                Min(query_time_ms),
                Max(query_time_ms),
                Avg(query_time_ms),
                Stddev(query_time_ms) 
            FROM
                query_logs
            WHERE
                subgraph = $1
                AND query_time_ms IS NOT NULL
            """,
            subgraph_ipfs_hash,
        )

        assert row

        logging.debug(
            "Subgraph average query stats for subgraph %s: %s", subgraph_ipfs_hash, row
        )

        return LogsDB.QueryStats(
            query="default",
            count=row[0],
            min_time=row[1],
            max_time=row[2],
            avg_time=float(row[3]),
            stddev_time=float(row[4]),
        )

    async def get_frequent_query_hashes_without_timing(self, min_count: int = 100):
        assert self.connection
        rows = await self.connection.fetch(
            """
            SELECT
                query_hash
            FROM
                query_logs
            GROUP BY
                query_hash
            HAVING
                Count(id) >= $1
                AND Sum(CASE WHEN query_time_ms IS NOT NULL THEN 1 ELSE 0 END) < $2
            """,
            min_count,
            min_count,
        )

        logging.debug("Frequent query hashes: %s", rows)

        return list(bytes(row[0]).hex() for row in rows)

    # def get_random_variables_for_query(self, query_hash: bytes):


if __name__ == "__main__":

    async def test():
        ldb = LogsDB()
        await ldb.connect()
        # r = await ldb.get_frequent_query_hashes_without_timing()
        r = await ldb.get_subgraph_average_query_stats(
            "Qmaz1R8vcv9v3gUfksqiS9JUz7K9G8S5By3JYn8kTiiP5K"
        )
        print(r)

    asyncio.run(test())
