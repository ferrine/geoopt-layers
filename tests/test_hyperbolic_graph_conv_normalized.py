import itertools
import torch
import pytest
from geoopt_layers.poincare import PoincareBallExact
from geoopt_layers.poincare.graph import HyperbolicGCNConv


@pytest.fixture(params=[True, False])
def disable1(request):
    return request.param


@pytest.fixture(params=[True, False])
def disable2(request):
    return request.param


@pytest.mark.parametrize(
    "aggr_method,bias,sizes,weighted,local",
    itertools.product(
        ["einstein", "tangent"],
        [True, False],
        [(5, 5), (5, 7)],
        [True, False],
        [True, False],
    ),
)
def test_graph_conv(aggr_method, bias, sizes, weighted, local, disable1, disable2):
    ball = PoincareBallExact(disable=disable1)
    ball_out = PoincareBallExact(c=0.1, disable=disable2)
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]])
    if weighted:
        edge_weight = torch.rand(edge_index.size(1))
    else:
        edge_weight = None
    x = ball.random(3, 5)
    out = HyperbolicGCNConv(
        *sizes,
        aggr_method=aggr_method,
        bias=bias,
        ball=ball,
        ball_out=ball_out,
        local=local
    )(x, edge_index, edge_weight=edge_weight)
    assert out.shape == (3, sizes[-1])
    ball_out.assert_check_point_on_manifold(out)
