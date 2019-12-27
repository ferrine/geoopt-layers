import torch
import numpy as np
import geoopt_layers
from geoopt_layers.poincare.graph.message_passing import HyperbolicMessagePassing
import pytest


@pytest.fixture(params=[True, False])
def disable(request):
    return request.param


def test_message_passing(disable):
    ball = geoopt_layers.poincare.PoincareBallExact(disable=disable)
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]])

    # x = torch.Tensor([[1], [2]])
    x = ball.random(2, 5)
    out = HyperbolicMessagePassing(flow="source_to_target", ball=ball).propagate(
        edge_index, x=x, size=(2, 3)
    )
    assert out.shape == (3, 5)
    mean = geoopt_layers.poincare.math.poincare_mean(x, ball=ball)
    np.testing.assert_allclose(out, mean.unsqueeze(0).expand_as(out), atol=1e-5)

    # x = torch.Tensor([[1], [2], [3]])
    x = ball.random(3, 5)
    out = HyperbolicMessagePassing(flow="source_to_target", ball=ball).propagate(
        edge_index, x=x
    )
    assert out.shape == (3, 5)
    ball.assert_check_point_on_manifold(out)

    # x = torch.Tensor([[1], [2], [3]])
    x = ball.random(3, 5)
    out = HyperbolicMessagePassing(flow="target_to_source", ball=ball).propagate(
        edge_index, x=x, size=(2, 3)
    )
    assert out.shape == (2, 5)
    ball.assert_check_point_on_manifold(out)

    # x = torch.Tensor([[1], [2], [3]])
    x = ball.random(3, 5)
    out = HyperbolicMessagePassing(flow="target_to_source", ball=ball).propagate(
        edge_index, x=x
    )
    assert out.shape == (3, 5)
    ball.assert_check_point_on_manifold(out)

    x = (ball.random(2, 5), ball.random(3, 5))
    out = HyperbolicMessagePassing(flow="source_to_target", ball=ball).propagate(
        edge_index, x=x
    )
    assert out.shape == (3, 5)
    ball.assert_check_point_on_manifold(out)

    x = (ball.random(2, 5), ball.random(3, 5))
    out = HyperbolicMessagePassing(flow="target_to_source", ball=ball).propagate(
        edge_index, x=x
    )
    assert out.shape == (2, 5)
    ball.assert_check_point_on_manifold(out)
