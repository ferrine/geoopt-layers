import itertools
import torch
import numpy as np
import pytest
import geoopt
from torch_geometric.nn.conv import GraphConv
from geoopt_layers.poincare.graph.graph_conv import HyperbolicGraphConv


@pytest.mark.parametrize(
    "aggr,bias,local,sizes,weighted",
    itertools.product(
        ["mean", "add"], [True, False], [True, False], [(5, 5), (5, 7)], [True, False],
    ),
)
def test_graph_conv(aggr, bias, local, sizes, weighted, ball_1, ball_2):
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]])
    if weighted:
        edge_weight = torch.rand(edge_index.size(1))
    else:
        edge_weight = None
    x = ball_1.random(3, 5)
    out = HyperbolicGraphConv(
        *sizes, aggr=aggr, local=local, ball=ball_1, ball_out=ball_2
    )(x, edge_index, size=(3, 3), edge_weight=edge_weight)
    assert out.shape == (3, sizes[-1])
    ball_2.assert_check_point_on_manifold(out)


@pytest.mark.parametrize(
    "aggr,bias,sizes,weighted",
    itertools.product(["mean", "add"], [True, False], [(5, 5), (5, 7)], [True, False],),
)
def test_graph_conv_matches_euclidean(aggr, bias, sizes, weighted):
    graph_conv = GraphConv(*sizes)
    ball = geoopt.Stereographic()
    hyp_graph_conv = HyperbolicGraphConv.from_graph_conv(graph_conv, ball=ball)
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]])
    if weighted:
        edge_weight = torch.rand(edge_index.size(1))
    else:
        edge_weight = None
    x = ball.random(3, 5)
    eucl_out = graph_conv(x, edge_index, size=(3, 3), edge_weight=edge_weight)
    assert eucl_out.shape == (3, sizes[-1])

    hyp_out = hyp_graph_conv(x, edge_index, size=(3, 3), edge_weight=edge_weight)
    assert hyp_out.shape == (3, sizes[-1])
    np.testing.assert_allclose(eucl_out.detach(), hyp_out.detach(), atol=1e-6)
