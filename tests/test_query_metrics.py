import asyncio
import autoagora.query_metrics
import vcr
from autoagora.config import init_config


class TestQueryMetrics:
    def test_subgraph_query_count(self):
        init_config(["--indexer-agent-mgmt-endpoint", "http://nowhere",
                     "--indexer-service-metrics-endpoint",
                     "http://indexer-service.default.svc.cluster.local:7300/metrics"])
        with vcr.use_cassette('vcr_cassettes/test_subgraph_query_count.yaml'):
            res = asyncio.run(
                autoagora.query_metrics.subgraph_query_count("Qmadj8x9km1YEyKmRnJ6EkC2zpJZFCfTyTZpuqC3j6e1QH")
            )
        print(res)
        assert (res == 938)

    def test_subgraph_query_count_multiple_endpoints(self):
        init_config(["--indexer-agent-mgmt-endpoint", "http://nowhere",
                     "--indexer-service-metrics-endpoint",
                     "http://indexer-service-0:7300/metrics,http://indexer-service-1:7300/metrics"])
        with vcr.use_cassette('vcr_cassettes/test_subgraph_query_count_multiple_endpoints.yaml'):
            res = asyncio.run(
                autoagora.query_metrics.subgraph_query_count("Qmadj8x9km1YEyKmRnJ6EkC2zpJZFCfTyTZpuqC3j6e1QH")
            )
        print(res)
        assert (res == 2607)

