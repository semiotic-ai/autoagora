K8S_EVENT = {
    "type": "ADDED",
    "object": {
        "api_version": "v1",
        "kind": "Endpoints",
        "metadata": {
            "annotations": {
                "endpoints.kubernetes.io/last-change-trigger-time": "2023-05-30T18:39:31Z"
            },
            "creation_timestamp": "2022-12-18T20:12:39Z",
            "deletion_grace_period_seconds": None,
            "deletion_timestamp": None,
            "finalizers": None,
            "generate_name": None,
            "generation": None,
            "labels": {"app": "indexer-service"},
            "managed_fields": None,
            "name": "indexer-service",
            "namespace": "namespace",
            "owner_references": None,
            "resource_version": "132261988",
            "self_link": None,
            "uid": "9305f611-5b3a-4ef0-94d9-ce7173f2c41b",
        },
        "subsets": [
            {
                "addresses": [
                    {
                        "hostname": None,
                        "ip": "192.168.42.78",
                        "node_name": "node-34",
                        "target_ref": {
                            "api_version": None,
                            "field_path": None,
                            "kind": "Pod",
                            "name": "indexer-service-64468444bf-bxbwz",
                            "namespace": "namespace",
                            "resource_version": None,
                            "uid": "1c40cdc0-d046-438a-a4cd-054b065dbc45",
                        },
                    },
                    {
                        "hostname": None,
                        "ip": "192.168.95.50",
                        "node_name": "node-1",
                        "target_ref": {
                            "api_version": None,
                            "field_path": None,
                            "kind": "Pod",
                            "name": "indexer-service-64468444bf-h98wp",
                            "namespace": "namespace",
                            "resource_version": None,
                            "uid": "b626c35c-130a-4baf-8ff4-e7f4bb9bc3dd",
                        },
                    },
                ],
                "not_ready_addresses": None,
                "ports": [
                    {
                        "app_protocol": None,
                        "name": "metrics",
                        "port": 7300,
                        "protocol": "TCP",
                    },
                    {
                        "app_protocol": None,
                        "name": "http",
                        "port": 7600,
                        "protocol": "TCP",
                    },
                ],
            }
        ],
    },
    "raw_object": {
        "kind": "Endpoints",
        "apiVersion": "v1",
        "metadata": {
            "name": "indexer-service",
            "namespace": "namespace",
            "uid": "9305f611-5b3a-4ef0-94d9-ce7173f2c41b",
            "resourceVersion": "132261988",
            "creationTimestamp": "2022-12-18T20:12:39Z",
            "labels": {"app": "indexer-service"},
            "annotations": {
                "endpoints.kubernetes.io/last-change-trigger-time": "2023-05-30T18:39:31Z"
            },
        },
        "subsets": [
            {
                "addresses": [
                    {
                        "ip": "192.168.42.78",
                        "nodeName": "node-34",
                        "targetRef": {
                            "kind": "Pod",
                            "namespace": "namespace",
                            "name": "indexer-service-64468444bf-bxbwz",
                            "uid": "1c40cdc0-d046-438a-a4cd-054b065dbc45",
                        },
                    },
                    {
                        "ip": "192.168.95.50",
                        "nodeName": "node-1",
                        "targetRef": {
                            "kind": "Pod",
                            "namespace": "namespace",
                            "name": "indexer-service-64468444bf-h98wp",
                            "uid": "b626c35c-130a-4baf-8ff4-e7f4bb9bc3dd",
                        },
                    },
                ],
                "ports": [
                    {"name": "metrics", "port": 7300, "protocol": "TCP"},
                    {"name": "http", "port": 7600, "protocol": "TCP"},
                ],
            }
        ],
    },
}
