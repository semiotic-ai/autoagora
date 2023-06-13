from unittest import mock

from easydict import EasyDict

from autoagora.k8s_service_watcher import K8SServiceEndpointsWatcher, aio
from tests.utils.constants import K8S_EVENT


class TestK8SServiceEndpointsWatcher:
    def test_k8s_service_creation(self):
        with mock.patch("builtins.open") as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = (
                "namespace"
            )
            with mock.patch(
                "autoagora.k8s_service_watcher.aio.ensure_future"
            ) as mock_ensure_future:
                with mock.patch(
                    "autoagora.k8s_service_watcher.K8SServiceEndpointsWatcher._watch_loop"
                ) as mock_watch_loop:

                    k8ssew = K8SServiceEndpointsWatcher("mock_service_name")
                    mock_watch_loop.assert_called_once()
                    mock_ensure_future.assert_called_once()

                    assert k8ssew.endpoint_ips == []
                    assert k8ssew._service_name == "mock_service_name"
                    assert k8ssew._namespace == "namespace"

    async def test_watch(self):

        loop = aio.new_event_loop()
        aio.set_event_loop(loop)
        event_mock = mock.MagicMock()
        event_mock.__getitem__.return_value = EasyDict(K8S_EVENT["object"])
        event_stream_mock = mock.MagicMock()
        event_stream_mock.__next__.return_value = event_mock
        w = mock.MagicMock()
        w.stream.return_value = event_stream_mock

        with mock.patch("builtins.open") as mock_open:
            with mock.patch(
                "autoagora.k8s_service_watcher.K8SServiceEndpointsWatcher._watch_loop"
            ) as mock_watch_loop:  # pyright: ignore[reportUnusedVariable]
                with mock.patch(
                    "autoagora.k8s_service_watcher.config.load_incluster_config"
                ) as mock_load_incluster_config:  # pyright: ignore[reportUnusedVariable]
                    with mock.patch(
                        "autoagora.k8s_service_watcher.watch.Watch",
                        return_value=w,
                    ) as mock_Watch_stream:  # pyright: ignore[reportUnusedVariable]
                        mock_open.return_value.__enter__.return_value.read.return_value = (
                            "namespace"
                        )

                        service_name = "my-service"
                        watcher = K8SServiceEndpointsWatcher(service_name)

                        task = aio.create_task(watcher._watch())
                        await aio.sleep(0.001)
                        task.cancel()
                        assert watcher.endpoint_ips == [
                            "192.168.42.78",
                            "192.168.95.50",
                        ]
                        aio.set_event_loop(None)
