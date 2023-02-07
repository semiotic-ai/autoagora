# Copyright 2023-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import asyncpg


@dataclass
class SaveState:
    last_update: datetime
    mean: float
    stddev: float


class PriceSaveStateDB:
    def __init__(self, pgpool: asyncpg.Pool) -> None:
        self.pgpool = pgpool
        self._table_created = False

    async def _create_table_if_not_exists(self) -> None:
        if not self._table_created:
            async with self.pgpool.acquire() as connection:
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

        async with self.pgpool.acquire() as connection:
            await connection.execute(
                """
                INSERT INTO price_save_state (subgraph, last_update, mean, stddev)
                    VALUES($1, $2, $3, $4)
                ON CONFLICT (subgraph)
                    DO
                    UPDATE SET
                        last_update = $2,
                        mean        = $3,
                        stddev      = $4
                """,
                subgraph,
                datetime.now(timezone.utc),
                mean,
                stddev,
            )

    async def load_state(self, subgraph: str) -> Optional[SaveState]:
        await self._create_table_if_not_exists()

        async with self.pgpool.acquire() as connection:
            row = await connection.fetchrow(
                """
                SELECT
                    last_update,
                    mean,
                    stddev
                FROM
                    price_save_state
                WHERE
                    subgraph = $1
                """,
                subgraph,
            )

        if row:
            return SaveState(
                last_update=row["last_update"],  # type: ignore
                mean=row["mean"],  # type: ignore
                stddev=row["stddev"],  # type: ignore
            )
