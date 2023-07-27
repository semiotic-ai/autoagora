# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Mapping, Optional

import aiohttp
import backoff
from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport

from autoagora.config import args


@backoff.on_exception(
    backoff.expo, aiohttp.ClientError, max_time=30, logger=logging.root
)
async def query_graph_node(query: str, variables: Optional[Mapping] = None):
    async with Client(
        transport=AIOHTTPTransport(args.graph_node_query_endpoint),
        fetch_schema_from_transport=False,
    ) as session:
        result = await session.execute(gql(query), variable_values=variables)  # type: ignore
    return result
