import asyncio
from unittest import mock

from autoagora.main import (
    DEFAULT_AGORA_VARIABLES,
    allocated_subgraph_watcher,
    init_config,
)


class TestScript:
    async def test_allocated_subgraph_watcher(self, postgresql):
        subgraph1 = "QmPnu3R7Fm4RmBF21aCYUohDmWbKd3VMXo64ACiRtwUQrn"
        subgraph2 = "QmTJBvvpknMow6n4YU8R9Swna6N8mHK8N2WufetysBiyuL"
        with mock.patch(
            "autoagora.logs_db.LogsDB.create_mrq_log_table"
        ) as mock_create_mrq_log_table:
            with mock.patch(
                "autoagora.main.get_allocated_subgraphs"
            ) as mock_get_allocated_subgraphs:
                with mock.patch("autoagora.main.set_cost_model") as mock_set_cost_model:
                    with mock.patch(
                        "autoagora.main.apply_default_model"
                    ) as mock_apply_default_model:  # pyright: ignore[reportUnusedVariable]
                        with mock.patch(
                            "autoagora.main.model_update_loop"
                        ) as mock_model_update_loop:
                            with mock.patch(
                                "autoagora.main.price_bandit_loop"
                            ) as mock_price_bandit_loop:
                                init_config(
                                    [
                                        "--indexer-agent-mgmt-endpoint",
                                        "http://nowhere",
                                        "--postgres-host",
                                        postgresql.info.host,
                                        "--postgres-username",
                                        postgresql.info.user,
                                        "--postgres-password",
                                        postgresql.info.password,
                                        "--postgres-port",
                                        str(postgresql.info.port),
                                        "--postgres-database",
                                        postgresql.info.dbname,
                                        "--indexer-service-metrics-endpoint",
                                        "http://indexer-service.default.svc.cluster.local:7300/metrics",
                                    ]
                                )
                                mock_create_mrq_log_table.return_value = None
                                mock_get_allocated_subgraphs.return_value = {
                                    subgraph1,
                                    subgraph2,
                                }

                                task = asyncio.create_task(allocated_subgraph_watcher())
                                await asyncio.sleep(2)
                                task.cancel()

                                mock_get_allocated_subgraphs.assert_called_once()
                                mock_set_cost_model.assert_any_call(
                                    subgraph1,
                                    variables=DEFAULT_AGORA_VARIABLES,
                                )
                                mock_set_cost_model.assert_any_call(
                                    subgraph2,
                                    variables=DEFAULT_AGORA_VARIABLES,
                                )
                                # Since there is no args for relative query cost the update_loop wont be called
                                assert mock_model_update_loop.call_count == 0
                                mock_price_bandit_loop.assert_called()
