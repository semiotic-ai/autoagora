# Copyright 2023-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import psycopg_pool
from psycopg import sql


@dataclass
class SaveState:
    last_update: datetime
    mean: float
    stddev: float


class PriceSaveStateDB:
    def __init__(self, pgpool: psycopg_pool.AsyncConnectionPool) -> None:
        self.pgpool = pgpool
        self._table_created = False

    async def _create_table_if_not_exists(self) -> None:
        if not self._table_created:
            async with self.pgpool.connection() as connection:
                await connection.execute(  # type: ignore
                    """
                    CREATE TABLE IF NOT EXISTS price_save_state (
                        subgraph        char(46)            PRIMARY KEY,
                        last_update     timestamptz         NOT NULL,
                        mean            double precision    NOT NULL,
                        stddev          double precision    NOT NULL
                    )
                    """
                )
            self._table_created = True

    async def save_state(self, subgraph: str, mean: float, stddev: float):
        await self._create_table_if_not_exists()

        async with self.pgpool.connection() as connection:
            await connection.execute(
                sql.SQL(
                    """
                INSERT INTO price_save_state (subgraph, last_update, mean, stddev)
                    VALUES({subgraph_hash}, {datetime}, {mean}, {stddev})
                ON CONFLICT (subgraph)
                    DO
                    UPDATE SET
                        last_update = {datetime},
                        mean        = {mean},
                        stddev      = {stddev}
                """
                ).format(
                    subgraph_hash=subgraph,
                    datetime=str(datetime.now(timezone.utc)),
                    mean=mean,
                    stddev=stddev,
                )
            )

    async def load_state(self, subgraph: str) -> Optional[SaveState]:
        await self._create_table_if_not_exists()

        async with self.pgpool.connection() as connection:
            row = await connection.execute(
                sql.SQL(
                    """
                SELECT
                    last_update,
                    mean,
                    stddev
                FROM
                    price_save_state
                WHERE
                    subgraph = {subgraph_hash}
                """
                ).format(subgraph_hash=subgraph)
            )
        row = await row.fetchone()
        if row:
            return SaveState(
                last_update=row[0],  # type: ignore
                mean=row[1],  # type: ignore
                stddev=row[2],  # type: ignore
            )
