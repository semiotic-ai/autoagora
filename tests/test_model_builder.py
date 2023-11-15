import os
import re
import tempfile
from unittest import mock

from autoagora.config import init_config
from autoagora.logs_db import LogsDB
from autoagora.model_builder import apply_default_model, build_template
from tests.utils.constants import TEST_MANUAL_AGORA_ENTRY, TEST_QUERY_1, TEST_QUERY_2


class TestModelBuilder:
    async def test_build_model(self, postgresql):
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
        most_frequent_queries = [
            LogsDB.QueryStats(
                query=TEST_QUERY_1,
                count=100,
                min_time=1,
                max_time=60,
                avg_time=1.2,
                stddev_time=0.5,
            ),
            LogsDB.QueryStats(
                query=TEST_QUERY_2,
                count=10,
                min_time=3,
                max_time=20,
                avg_time=13.2,
                stddev_time=0.7,
            ),
        ]

        model = build_template(
            "Qmadj8x9km1YEyKmRnJ6EkC2zpJZFCfTyTZpuqC3j6e1QH", most_frequent_queries
        )
        pattern = r"# Generated by AutoAgora \d+\.\d+\.\d+"
        # To ensure a version is being obtained
        assert re.match(pattern, model), f"{model} does not match pattern {pattern}"
        assert TEST_QUERY_1 in model
        assert TEST_QUERY_2 in model
        assert TEST_MANUAL_AGORA_ENTRY not in model

    async def test_build_model_with_manual_entry(self, postgresql):
        subgraph = "Qmadj8x9km1YEyKmRnJ6EkC2zpJZFCfTyTZpuqC3j6e1QH"
        file_type = ".agora"
        # Creates a temp dir to simulate manual entries
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a file inside temp dir which will be deleted once out of context
            with open(os.path.join(temp_dir, subgraph + file_type), "w") as temp_file:
                temp_file.write(TEST_MANUAL_AGORA_ENTRY)
            most_frequent_queries = [
                LogsDB.QueryStats(
                    query=TEST_QUERY_1,
                    count=100,
                    min_time=1,
                    max_time=60,
                    avg_time=1.2,
                    stddev_time=0.5,
                ),
                LogsDB.QueryStats(
                    query=TEST_QUERY_2,
                    count=10,
                    min_time=3,
                    max_time=20,
                    avg_time=13.2,
                    stddev_time=0.7,
                ),
            ]
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
                    "--manual-entry-path",
                    temp_dir,
                ]
            )
            model = build_template(subgraph, most_frequent_queries)
            pattern = r"# Generated by AutoAgora \d+\.\d+\.\d+"
            # To ensure a version is being obtained
            assert re.match(pattern, model), f"{model} does not match pattern {pattern}"
            assert TEST_QUERY_1 in model
            assert TEST_QUERY_2 in model
            assert TEST_MANUAL_AGORA_ENTRY in model

    async def test_apply_default_model(self, postgresql):
        subgraph = "Qmadj8x9km1YEyKmRnJ6EkC2zpJZFCfTyTZpuqC3j6e1QH"
        file_type = ".agora"
        # Creates a temp dir to simulate manual entries
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a file inside temp dir which will be deleted once out of context
            with open(os.path.join(temp_dir, subgraph + file_type), "w") as temp_file:
                temp_file.write(TEST_MANUAL_AGORA_ENTRY)
            with mock.patch(
                "autoagora.model_builder.set_cost_model"
            ) as set_cost_model_mock:
                set_cost_model_mock.return_value = None
                with mock.patch("logging.debug") as logging_debug_mock:
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
                            "--manual-entry-path",
                            temp_dir,
                        ]
                    )
                    await apply_default_model(subgraph)
                    # Obtain the args send to the logger.debug fn
                    debug_logs_args = logging_debug_mock.call_args[0]
                    # Remove \n since they cause the assertion to bug and fail
                    debug_logs_args = str(debug_logs_args).replace(r"\n", "")
                    manual_agora_entry = TEST_MANUAL_AGORA_ENTRY.replace("\n", "")
                    assert manual_agora_entry in debug_logs_args
