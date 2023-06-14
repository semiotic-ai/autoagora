# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

import asyncpg
import graphql


class LogsDB:
    @dataclass
    class QueryStats:
        query: str
        count: int
        min_time: int
        max_time: int
        avg_time: float
        stddev_time: float

    def __init__(self, pgpool: asyncpg.Pool) -> None:
        self.pgpool = pgpool

    def return_query_body(self, query):
        # Keep only query body -- ie. no var defs
        query = graphql.parse(query)
        assert len(query.definitions) == 1  # Should be single root query
        query = query.definitions[0].selection_set  # type: ignore
        query = "query " + graphql.print_ast(query)
        return query

    async def get_most_frequent_queries(
        self, subgraph_ipfs_hash: str, min_count: int = 100
    ):
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
