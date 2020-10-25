import itertools
import torch
import pytest
from geoopt_layers.poincare.graph import HyperbolicGCNConv


@pytest.mark.parametrize(
    "sizes,weighted,n_in,n_out,features",
    itertools.product(
        [(5, 5), (5, 7)], [True, False], [1, 2], [1, 2], ["hyperplanes", "gromov"]
    ),
)
def test_graph_conv(sizes, weighted, ball_1, ball_2, n_in, n_out, features):
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]])
    if weighted:
        edge_weight = torch.rand(edge_index.size(1))
    else:
        edge_weight = None
    x = ball_1.random(n_in, 3, 5)
    x = torch.cat(x.unbind(0), -1)
    layer = HyperbolicGCNConv(
        *sizes,
        num_basis=sizes[1] * 2,
        balls=[ball_1] * n_in,
        balls_out=[ball_2] * n_out,
        features=features,
    )
    out = layer(x, edge_index, edge_weight=edge_weight)
    assert out.shape == (3, sizes[-1] * n_out)
    for x in out.chunk(n_out, -1):
        ball_2.assert_check_point_on_manifold(x)
