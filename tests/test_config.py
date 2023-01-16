import pytest

from autoagora.config import args, init_config


class TestConfigArgs:
    """Testing the command line arguments handling.

    Testing because there's some not so trivial stuff going on with the optional
    `--experimental-model-builder`.
    """

    def test_model_builder_off(self):
        indexer_mgmt_endpoint_value = "http://indexer-something:1234"
        indexer_service_metrics_endpoint_value = "https://indexer-service:2345"

        init_config(
            [
                "--indexer-agent-mgmt-endpoint",
                indexer_mgmt_endpoint_value,
                "--indexer-service-metrics-endpoint",
                indexer_service_metrics_endpoint_value,
            ]
        )

        # Expecting no errors as we've provided all mandatory arguments in the case
        # where the `--experimental-model-builder` is not enabled

        assert args.indexer_agent_mgmt_endpoint == indexer_mgmt_endpoint_value
        assert (
            args.indexer_service_metrics_endpoint
            == indexer_service_metrics_endpoint_value
        )

    def test_model_builder_on_missing_args(self):
        indexer_mgmt_endpoint_value = "http://indexer-something:1234"
        indexer_service_metrics_endpoint_value = "https://indexer-service:2345"

        # Expecting errors because enabling the `--experimental-model-builder` adds new
        # mandatory arguments
        with pytest.raises(SystemExit) as exit_error:
            init_config(
                [
                    "--indexer-agent-mgmt-endpoint",
                    indexer_mgmt_endpoint_value,
                    "--indexer-service-metrics-endpoint",
                    indexer_service_metrics_endpoint_value,
                    "--experimental-model-builder",
                ]
            )

        assert exit_error.value.code == 2

    def test_model_builder_on(self):
        # These are always mandatory
        indexer_mgmt_endpoint_value = "http://indexer-something:1234"
        indexer_service_metrics_endpoint_value = "https://indexer-service:2345"

        # These are mandatory when `--experimental-model-builder` is on
        logs_postgres_host_value = "pghost"
        logs_postgres_database_value = "pgdatabase"
        logs_postgres_username_value = "pguser"
        logs_postgres_password_value = "pgpass"

        init_config(
            [
                "--indexer-agent-mgmt-endpoint",
                indexer_mgmt_endpoint_value,
                "--indexer-service-metrics-endpoint",
                indexer_service_metrics_endpoint_value,
                "--experimental-model-builder",
                "--logs-postgres-host",
                logs_postgres_host_value,
                "--logs-postgres-database",
                logs_postgres_database_value,
                "--logs-postgres-username",
                logs_postgres_username_value,
                "--logs-postgres-password",
                logs_postgres_password_value,
            ]
        )

        assert args.indexer_agent_mgmt_endpoint == indexer_mgmt_endpoint_value
        assert (
            args.indexer_service_metrics_endpoint
            == indexer_service_metrics_endpoint_value
        )
        assert args.logs_postgres_host == logs_postgres_host_value
        assert args.logs_postgres_database == logs_postgres_database_value
        assert args.logs_postgres_username == logs_postgres_username_value
        assert args.logs_postgres_password == logs_postgres_password_value
