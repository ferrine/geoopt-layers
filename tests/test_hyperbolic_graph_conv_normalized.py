import itertools
import torch
import pytest
import geoopt
import numpy as np
from torch_geometric.nn.conv import GCNConv
from geoopt_layers.poincare.graph import HyperbolicGCNConv


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
def test_graph_conv(aggr_method, bias, sizes, weighted, local, ball_1, ball_2):
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]])
    if weighted:
        edge_weight = torch.rand(edge_index.size(1))
    else:
        edge_weight = None
    x = ball_1.random(3, 5)
    out = HyperbolicGCNConv(*sizes, ball=ball_1, ball_out=ball_2, local=local)(
        x, edge_index, edge_weight=edge_weight
    )
    assert out.shape == (3, sizes[-1])
    ball_2.assert_check_point_on_manifold(out)


@pytest.mark.parametrize(
    "bias,sizes,weighted,normalize,improved",
    itertools.product(
        [True, False], [(5, 5), (5, 7)], [True, False], [True, False], [True, False]
    ),
)
def test_graph_conv_matches_euclidean(bias, sizes, weighted, normalize, improved):
    graph_conv = GCNConv(*sizes, bias=bias, normalize=normalize, improved=improved)
    if bias:
        with torch.no_grad():
            graph_conv.bias.normal_()
    ball = geoopt.Stereographic()
    hyp_graph_conv = HyperbolicGCNConv.from_gcn_conv(graph_conv, ball=ball)
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]])
    if weighted:
        edge_weight = torch.rand(edge_index.size(1))
    else:
        edge_weight = None
    x = ball.random(3, 5)
    eucl_out = graph_conv(x, edge_index, edge_weight=edge_weight)
    assert eucl_out.shape == (3, sizes[-1])

    hyp_out = hyp_graph_conv(x, edge_index, edge_weight=edge_weight)
    assert hyp_out.shape == (3, sizes[-1])
    np.testing.assert_allclose(eucl_out.detach(), hyp_out.detach(), atol=1e-6)
