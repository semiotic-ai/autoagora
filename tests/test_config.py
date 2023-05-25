from argparse import Namespace

from autoagora.config import args, init_config


class TestInitConfig:
    def test_default_arguments(self):
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
        # Testing default values
        assert args.log_level == "WARNING"
        assert args.postgres_database == "autoagora"
        assert args.postgres_port == 5432
        # Testing given values
        assert args.indexer_agent_mgmt_endpoint == "http://nowhere"
        assert args.postgres_host == "nowhere"
        assert args.postgres_username == "nowhere"
        assert args.postgres_password == "nowhere"
