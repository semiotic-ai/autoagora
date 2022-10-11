# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from numbers import Number
from typing import Any, Dict, Mapping, Optional, Set

import aiohttp
import backoff
import configargparse
from base58 import b58decode, b58encode
from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport

from autoagora.config import args

argsparser = configargparse.get_argument_parser()
argsparser.add_argument(
    "--indexer-agent-mgmt-endpoint",
    env_var="INDEXER_AGENT_MGMT_ENDPOINT",
    required=True,
    help="URL to the indexer-agent management GraphQL endpoint.",
)


def ipfs_hash_to_hex(ipfs_hash: str) -> str:
    assert len(ipfs_hash) == 46
    assert ipfs_hash.startswith("Qm")

    # Remove the leading Qm/0x1220
    return "0x" + b58decode(ipfs_hash)[2:].hex()


def hex_to_ipfs_hash(hex: str) -> str:
    if hex.startswith("0x"):
        hex = hex[:2]
    assert len(hex) == 64
    hex = "1220" + hex  # Prepend 1220
    return str(b58encode(hex))


@backoff.on_exception(
    backoff.expo, aiohttp.ClientError, max_time=30, logger=logging.root
)
async def query_indexer_agent(query: str, variables: Optional[Mapping] = None):
    async with Client(
        transport=AIOHTTPTransport(args.indexer_agent_mgmt_endpoint),
        fetch_schema_from_transport=False,
    ) as session:
        result = await session.execute(gql(query), variable_values=variables)  # type: ignore
    return result


async def get_indexed_subgraphs() -> Set[str]:
    result = await query_indexer_agent(
        """
        {
            indexerDeployments{
                subgraphDeployment
            }
        }
        """
    )

    return set(e["indexerDeployments"] for e in result["indexerAllocations"])


async def get_allocated_subgraphs() -> Set[str]:
    result = await query_indexer_agent(
        """
        {
            indexerAllocations{
                subgraphDeployment
            }
        }
        """
    )

    return set(e["subgraphDeployment"] for e in result["indexerAllocations"])


async def set_cost_model(
    subgraph: str, model: Optional[str] = None, variables: Optional[Mapping] = None
):
    """Send Agora cost model and/or variables for a subgraph to the indexer-agent.

    Can send only the model document or the variables. The indexer-agent will ignore
    `None` values.

    Args:
        subgraph (str): Subgraph IPFS hash.
        model (Optional[str], optional): Agora model document. Defaults to None.
        variables (Optional[Mapping], optional): Agora model variables. Defaults to
            None.

    Raises:
        RuntimeError: `model` and `variables` are both `None`.
    """

    if model is None and variables is None:
        raise RuntimeError("'model' and 'variables' arguments are both None.")

    if variables is not None:
        # Format numbers
        variables = {
            k: f"{v:.18f}" if isinstance(v, Number) else v for k, v in variables.items()
        }
    variables_json = json.dumps(variables)

    await query_indexer_agent(
        """
        mutation ($deployment: String!, $model: String, $variables: String) {
            setCostModel(
                costModel: {
                    deployment: $deployment,
                    model: $model,
                    variables: $variables
                }
            ) {
                __typename
            }
        }
        """,
        variables={
            "deployment": ipfs_hash_to_hex(subgraph),
            "model": model,
            "variables": variables_json,
        },
    )


async def get_cost_model(subgraph: str) -> str:
    result = await query_indexer_agent(
        """
        query ($deployment: String!){
            costModel(deployment: $deployment) {
                model
            }
        }
        """,
        variables={
            "deployment": ipfs_hash_to_hex(subgraph),
        },
    )

    return result["costModel"]["model"]


async def get_cost_variables(subgraph: str) -> Dict[str, Any]:
    result = await query_indexer_agent(
        """
        query ($deployment: String!){
            costModel(deployment: $deployment) {
                variables
            }
        }
        """,
        variables={
            "deployment": ipfs_hash_to_hex(subgraph),
        },
    )

    return json.loads(result["costModel"]["variables"])
