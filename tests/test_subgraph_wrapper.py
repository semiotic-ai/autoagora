from unittest import mock
from unittest.mock import patch

from autoagora.subgraph_wrapper import (
    MetricsEndpoints,
    SubgraphWrapper,
    get_cost_variables,
    set_cost_model,
    subgraph_query_count,
    time,
)


class TestSubgraphWrapper:
    async def test_set_cost_multiplier(self):
        subgraph = "QmTJBvvpknMow6n4YU8R9Swna6N8mHK8N2WufetysBiyuL"
        cost_multiplier = 1.5
        subgraph_wrapper = SubgraphWrapper(subgraph)
        with mock.patch(
            "autoagora.subgraph_wrapper.get_cost_variables"
        ) as mock_get_cost_variables:
            mock_get_cost_variables.return_value = {"GLOBAL_COST_MULTIPLIER": 1.0}
            with mock.patch(
                "autoagora.subgraph_wrapper.set_cost_model"
            ) as mock_set_cost_model:
                with mock.patch("autoagora.subgraph_wrapper.time") as mock_time:

                    mock_time.return_value = 10003.0

                    await subgraph_wrapper.set_cost_multiplier(cost_multiplier)

                    mock_get_cost_variables.assert_called_once_with(subgraph)
                    mock_set_cost_model.assert_called_once_with(
                        subgraph, variables={"GLOBAL_COST_MULTIPLIER": cost_multiplier}
                    )
                    assert subgraph_wrapper.last_change_time == 10003.0

    async def test_queries_per_second(self):
        subgraph = "QmTJBvvpknMow6n4YU8R9Swna6N8mHK8N2WufetysBiyuL"
        cost_multiplier = 1.5
        subgraph_wrapper = SubgraphWrapper(subgraph)
        with mock.patch(
            "autoagora.subgraph_wrapper.MetricsEndpoints"
        ) as mock_metric_endpoints_class:
            mock_metric_endpoint_obj = mock_metric_endpoints_class.return_value
            with mock.patch(
                "autoagora.subgraph_wrapper.subgraph_query_count"
            ) as mock_subgraph_query_count:
                with mock.patch("autoagora.subgraph_wrapper.time") as mock_time:

                    mock_time.side_effect = [4, 8]
                    mock_subgraph_query_count.side_effect = [2, 6]

                    qps = await subgraph_wrapper.queries_per_second(
                        mock_metric_endpoint_obj, 0.1
                    )

                    mock_subgraph_query_count.assert_called_with(
                        subgraph, mock_metric_endpoint_obj
                    )
                    assert mock_subgraph_query_count.call_count == 2
                    assert qps == 1
