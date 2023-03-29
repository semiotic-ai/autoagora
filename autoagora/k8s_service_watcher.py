# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import asyncio as aio
import logging

from kubernetes import client, config, watch
from kubernetes.client.api_client import ApiClient
from kubernetes.client.rest import ApiException

from autoagora.misc import async_exit_on_exception


class K8SServiceEndpointsWatcher:
    def __init__(self, service_name: str) -> None:
        """Maintains an automatically, asynchronously updated list of endpoints backing
        a kubernetes service in the current namespace.

        This is supposed to be run from within a Kubernetes pod. The pod will need a
        role that grants it:

        ```
            rules:
            - apiGroups: [""]
                resources: ["endpoints"]
                verbs: ["watch"]
        ```

        Args:
            service_name (str): Kubernetes service name.

        Raises:
            FileNotFoundError: couldn't find
            `/var/run/secrets/kubernetes.io/serviceaccount/namespace`, which is
            expected when running within a Kubernetes pod container.
        """
        self.endpoint_ips = []
        self._service_name = service_name

        try:
            with open(
                "/var/run/secrets/kubernetes.io/serviceaccount/namespace", "r"
            ) as f:
                self._namespace = f.read().strip()
        except FileNotFoundError:
            logging.exception("Probably not running in Kubernetes.")
            raise

        # Starts the async _loop immediately
        self._future = aio.ensure_future(self._watch_loop())

    @async_exit_on_exception()
    async def _watch_loop(self) -> None:
        """Restarts the k8s watch on expiration."""
        while True:
            try:
                await self._watch()
            except ApiException as api_exc:
                if api_exc.status == watch.watch.HTTP_STATUS_GONE:
                    logging.debug("k8s_service_watcher 410 timeout.")
                else:
                    raise
            logging.debug("k8s_service_watcher restarted")

    async def _watch(self) -> None:
        """Watches for changes in k8s service endpoints."""
        config.load_incluster_config()

        api = ApiClient()
        v1 = client.CoreV1Api(api)
        w = watch.Watch()
        event_stream = w.stream(
            v1.list_namespaced_endpoints,
            namespace=self._namespace,
            field_selector=f"metadata.name={self._service_name}",
        )

        loop = aio.get_running_loop()

        while event := await loop.run_in_executor(None, next, event_stream):
            result = event["object"]  # type: ignore

            self.endpoint_ips = [
                address.ip
                for subset in result.subsets  # type: ignore
                for address in subset.addresses  # type: ignore
            ]

            logging.debug(
                "Got endpoint IPs for service %s: %s",
                self._service_name,
                self.endpoint_ips,
            )
