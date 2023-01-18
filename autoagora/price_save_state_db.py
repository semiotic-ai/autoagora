# Copyright 2023-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import asyncpg

from autoagora.config import args


@dataclass
class SaveState:
    last_update: datetime
    mean: float
    stddev: float


class PriceSaveStateDB:
    def __init__(self) -> None:
        self.connection: Optional[asyncpg.connection.Connection] = None

    async def connect(self) -> None:
        self.connection = await asyncpg.connect(  # type: ignore
            host=args.postgres_host,
            database=args.postgres_database,
            user=args.postgres_username,
            password=args.postgres_password,
            port=args.postgres_port,
        )

        await self.connection.execute(  # type: ignore
            """
            CREATE TABLE IF NOT EXISTS price_save_state (
                subgraph        char(46)            PRIMARY KEY,
                last_update     timestamptz         NOT NULL,
                mean            double precision    NOT NULL,
                stddev          double precision    NOT NULL
            )
            """
        )

    async def save_state(self, subgraph: str, mean: float, stddev: float):
        assert self.connection

        await self.connection.execute(
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
        assert self.connection

        row = await self.connection.fetchrow(
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
