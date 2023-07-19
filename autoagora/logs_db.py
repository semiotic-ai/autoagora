# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from datetime import datetime

import graphql
import psycopg_pool
from psycopg import sql


class LogsDB:
    @dataclass
    class QueryStats:
        query: str
        count: int
        min_time: int
        max_time: int
        avg_time: float
        stddev_time: float
 
    @dataclass
    class MRQ_Info:
        hash: bytes
        query: str
        query_time_ms: int = 0
        timestamp: datetime = datetime.now()

    def __init__(self, pgpool: psycopg_pool.AsyncConnectionPool) -> None:
        self.pgpool = pgpool


    async def create_mrq_log_table(self) -> None:
        async with self.pgpool.acquire() as connection:
            await connection.execute(
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

    def return_query_body(self, query):
        # Keep only query body -- ie. no var defs
        query = graphql.parse(query)
        assert len(query.definitions) == 1  # Should be single root query
        query = query.definitions[0].selection_set  # type: ignore
        query = "query " + graphql.print_ast(query)
        return query

    async def get_most_frequent_queries_null_time(
        self, subgraph_ipfs_hash: str, min_count: int = 100
    ):

        async with self.pgpool.connection() as connection:
            rows = await connection.execute(
                sql.SQL(
                    """
                SELECT
                    hash,
                    query
                    
                FROM
                    query_skeletons
                INNER JOIN
                (
                    SELECT
                        query_hash as qhash,
                        count(id) as count_id
                    FROM
                        query_logs
                    WHERE
                        subgraph = $1
                        AND query_time_ms IS NULL
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
            ))
            return [
                LogsDB.MRQ_Info(
                    hash=row[0],
                    query=self.return_query_body(row[1])
                    if self.return_query_body(row[1])
                    else "null",
                )
                for row in rows
            ]

    async def get_query_variables_from_query_hash(self, hash):
        async with self.pgpool.acquire() as connection:
            query_variables = await connection.fetch(
                """
                SELECT
                    query_variables
                FROM 
                    query_logs
	            WHERE
                    query_hash = $1;
                """,
                hash,
            )
            return query_variables

    async def save_generated_aa_query_values(
        self, query: MRQ_Info, subgraph: str, query_variables
    ):
        async with self.pgpool.acquire() as connection:
            query_variables = await connection.execute(
                """
                    INSERT INTO mrq_query_logs (
                            subgraph,
                            query_hash,
                            timestamp,
                            query_time_ms,
                            query_variables
                        )
                        VALUES (%s, %s, %s, %s, %s)
                    """,
                (
                    subgraph,
                    query.hash,
                    query.timestamp,
                    query.query_time_ms,
                    query_variables,
                ),
            )

    async def get_most_frequent_queries(
        self, subgraph_ipfs_hash: str, min_count: int = 100, mrq_table: bool = False
    ):
        most_frequent_query_table = "query_logs"
        if mrq_table:
            most_frequent_query_table = "mrq_query_logs"
        async with self.pgpool.acquire() as connection:
            rows = await connection.fetch(
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
                        $1
                    WHERE
                        subgraph = $2
                        AND query_time_ms IS NOT NULL
                    GROUP BY
                        qhash
                    HAVING
                        Count(id) >= $3
                ) as query_logs
                ON
                    qhash = hash
                ORDER BY
                    count_id DESC
                """,
                most_frequent_query_table,
                subgraph_ipfs_hash,
                min_count,
            )
        rows = await rows.fetchall()
        return [
            LogsDB.QueryStats(
                query=self.return_query_body(row[0])
                if self.return_query_body(row[0])
                else "null",
                count=row[1],
                min_time=row[2],
                max_time=row[3],
                avg_time=float(row[4]),
                stddev_time=float(row[5]),
            )
            for row in rows
        ]
