import itertools
import torch
import pytest
import geoopt
from geoopt_layers.poincare.graph.graph_conv import HyperbolicGraphConv


@pytest.mark.parametrize(
    "aggr,aggr_method,bias,local,sizes,weighted",
    itertools.product(
        ["mean", "sum"],
        ["einstein", "tangent"],
        [True, False],
        [True, False],
        [(5, 5), (5, 7)],
        [True, False],
    ),
)
def test_graph_conv(aggr, aggr_method, bias, local, sizes, weighted):
    ball = geoopt.PoincareBallExact()
    ball_out = geoopt.PoincareBallExact(c=0.1)
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]])
    if weighted:
        edge_weight = torch.rand(edge_index.size(1))
    else:
        edge_weight = None
    x = ball.random(3, 5)
    out = HyperbolicGraphConv(
        *sizes,
        aggr=aggr,
        aggr_method=aggr_method,
        bias=bias,
        local=local,
        ball=ball,
        ball_out=ball_out
    )(x, edge_index, size=(3, 3), edge_weight=edge_weight)
    assert out.shape == (3, sizes[-1])
    ball_out.assert_check_point_on_manifold(out)
