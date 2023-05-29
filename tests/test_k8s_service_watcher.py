#import asyncio
import vcr
from unittest import mock
import autoagora.query_metrics
from autoagora.config import args, init_config
from autoagora.k8s_service_watcher import K8SServiceEndpointsWatcher, aio


class TestK8SServiceEndpointsWatcher:
    def test_k8s_service_creation(self):
        with mock.patch("builtins.open") as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = "namespace"
            with mock.patch("autoagora.k8s_service_watcher.aio.ensure_future") as mock_ensure_future:
                with mock.patch(
                    "autoagora.k8s_service_watcher.K8SServiceEndpointsWatcher._watch_loop"
                ) as mock_watch_loop:

                    k8ssew = K8SServiceEndpointsWatcher("mock_service_name")
                    mock_watch_loop.assert_called_once()
                    mock_ensure_future.assert_called_once()

                    assert k8ssew.endpoint_ips == []
                    assert k8ssew._service_name == "mock_service_name"
                    assert k8ssew._namespace == "namespace"
