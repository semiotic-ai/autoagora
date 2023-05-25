from unittest import mock

from autoagora.config import args, init_config
from autoagora.indexer_utils import (
    get_allocated_subgraphs,
    get_cost_variables,
    hex_to_ipfs_hash,
    ipfs_hash_to_hex,
    query_indexer_agent,
    set_cost_model,
)


class TestHexIpfs:
    def test_hex_ipfs(self):
        hex = "0xbbde25a2c85f55b53b7698b9476610c3d1202d88870e66502ab0076b7218f98a"
        ipfs = hex_to_ipfs_hash(hex)
        assert ipfs == "Qmaz1R8vcv9v3gUfksqiS9JUz7K9G8S5By3JYn8kTiiP5K"

    def test_ipfs_hex(self):
        ipfs = "Qmaz1R8vcv9v3gUfksqiS9JUz7K9G8S5By3JYn8kTiiP5K"
        hex = ipfs_hash_to_hex(ipfs)
        assert (
            hex == "0xbbde25a2c85f55b53b7698b9476610c3d1202d88870e66502ab0076b7218f98a"
        )

    async def test_query_indexer_agent(self):
        init_config(
            [
                "--indexer-agent-mgmt-endpoint",
                "http://nowhere",
                "--postgres-host",
                "nowhere",
                "--postgres-username",
                "nowhere",
                "--postgres-password",
                "nowhere",
                "--indexer-service-metrics-endpoint",
                "http://indexer-service.default.svc.cluster.local:7300/metrics",
            ]
        )
        query = "{ indexerAllocations{ subgraphDeployment } }"
        with mock.patch("autoagora.indexer_utils.Client") as client_mock:
            mock_client = client_mock.return_value
            mock_client.execute.return_value = {
                "mockResponse": [
                    {
                        "subgraphDeployment": "QmUVskWrz1ZiQZ76AtyhcfFDEH1ELnRpoyEhVL8p6NFTbR"
                    },
                    {
                        "subgraphDeployment": "QmcBSr5R3K2M5tk8qeHFaX8pxAhdViYhcKD8ZegYuTcUhC"
                    },
                ]
            }
            res = await query_indexer_agent(query)
            assert res

    async def test_get_allocated_subgraphs(self):
        with mock.patch("autoagora.indexer_utils.query_indexer_agent") as qia_mock:
            qia_mock.return_value = {
                "indexerAllocations": [
                    {
                        "subgraphDeployment": "QmUVskWrz1ZiQZ76AtyhcfFDEH1ELnRpoyEhVL8p6NFTbR"
                    },
                    {
                        "subgraphDeployment": "QmcBSr5R3K2M5tk8qeHFaX8pxAhdViYhcKD8ZegYuTcUhC"
                    },
                ]
            }
            res = await get_allocated_subgraphs()
            assert res

    async def test_set_cost_model_success(self):
        subgraph = "QmUVskWrz1ZiQZ76AtyhcfFDEH1ELnRpoyEhVL8p6NFTbR"
        variables = {
            "DAI": "7.96129355477008894",
            "DEFAULT_COST": "50.000000000000000000",
            "GLOBAL_COST_MULTIPLIER": "0.000000063439006453",
        }
        with mock.patch("autoagora.indexer_utils.query_indexer_agent") as qia_mock:
            await set_cost_model(subgraph, variables=variables)
            qia_mock.assert_called_once()

    async def test_set_cost_model_failure(self):
        subgraph = "QmUVskWrz1ZiQZ76AtyhcfFDEH1ELnRpoyEhVL8p6NFTbR"
        error = None
        with mock.patch("autoagora.indexer_utils.query_indexer_agent") as qia_mock:
            try:
                await set_cost_model(subgraph)
            except Exception as e:
                error = e
            assert qia_mock.call_count == 0
            assert error

    async def test_get_cost_models(self):
        subgraph = "QmUVskWrz1ZiQZ76AtyhcfFDEH1ELnRpoyEhVL8p6NFTbR"
        with mock.patch("autoagora.indexer_utils.query_indexer_agent") as qia_mock:
            qia_mock.return_value = {
                "costModel": {
                    "variables": '{"DAI":"7.96129355477008894","DEFAULT_COST":"50.000000000000000000","GLOBAL_COST_MULTIPLIER":"0.000000063439006453"}'
                }
            }
            res = await get_cost_variables(subgraph)
            assert res
