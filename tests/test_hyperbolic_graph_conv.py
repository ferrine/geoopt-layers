import itertools
import torch
import pytest
import geoopt
from geoopt_layers.poincare.graph.graph_conv import HyperbolicGraphConv


@pytest.mark.parametrize(
    "aggr,aggr_method,bias,learn_origin,sizes",
    itertools.product(
        ["mean", "sum"],
        ["einstein", "tangent"],
        [True, False],
        [True, False],
        [(5, 5), (5, 7)],
    ),
)
def test_graph_conv(aggr, aggr_method, bias, learn_origin, sizes):
    ball = geoopt.PoincareBallExact()
    ball_out = geoopt.PoincareBallExact()
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]])

    x = ball.random(3, 5)
    out = HyperbolicGraphConv(
        *sizes,
        aggr=aggr,
        aggr_method=aggr_method,
        bias=bias,
        learn_origin=learn_origin,
        ball=ball,
        ball_out=ball_out
    )(x, edge_index, size=(3, 3))
    assert out.shape == (3, sizes[-1])
    ball.assert_check_point_on_manifold(out)
