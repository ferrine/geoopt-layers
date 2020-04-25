import itertools
import torch
import pytest
import numpy as np
import geoopt
from torch_geometric.nn.conv.spline_conv import SplineConv
from geoopt_layers.poincare.graph import HyperbolicSplineConv


@pytest.mark.parametrize(
    "bias,sizes,degree,kernel_size,root_weight,dim",
    itertools.product(
        [True, False],  # bias
        [(5, 5), (5, 7)],  # sizes
        [1, 2],  # degree
        [1, 2],  # kernel_size
        [True, False],  # root_weight
        [1, 2],  # dim
    ),
)
def test_spline_conv_1(
    bias, sizes, kernel_size, degree, root_weight, dim, ball_1, ball_2,
):
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3, 0], [1, 2, 3, 0, 0, 0, 0]])
    x = ball_1.random(4, 5)
    pseudo = torch.rand(edge_index.size(1), dim)
    out = HyperbolicSplineConv(
        *sizes,
        bias=bias,
        ball=ball_1,
        ball_out=ball_2,
        kernel_size=kernel_size,
        degree=degree,
        root_weight=root_weight,
        dim=dim,
    )(x, edge_index, pseudo=pseudo)
    assert out.shape == (4, sizes[-1])
    ball_2.assert_check_point_on_manifold(out)
    out.sum().backward()


@pytest.mark.parametrize(
    "bias,sizes,degree,kernel_size,root_weight,dim",
    itertools.product(
        [True, False],  # bias
        [(5, 5), (5, 7)],  # sizes
        [1, 2],  # degree
        [1, 2],  # kernel_size
        [True, False],  # root_weight
        [1, 2],  # dim
    ),
)
def test_spline_conv(bias, sizes, kernel_size, degree, root_weight, dim):
    ball = geoopt.Stereographic()
    ball_out = geoopt.Stereographic()
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3, 0], [1, 2, 3, 0, 0, 0, 0]])
    x = ball.random(4, 5)
    pseudo = torch.rand(edge_index.size(1), dim)
    new = HyperbolicSplineConv(
        *sizes,
        bias=bias,
        ball=ball,
        ball_out=ball_out,
        kernel_size=kernel_size,
        degree=degree,
        root_weight=root_weight,
        dim=dim,
    )

    orig = SplineConv(
        *sizes,
        bias=bias,
        kernel_size=kernel_size,
        degree=degree,
        root_weight=root_weight,
        dim=dim,
    )
    orig.load_state_dict(new.state_dict(), strict=False)
    x_new = new(x, edge_index, pseudo=pseudo)
    x_orig = orig(x, edge_index, pseudo=pseudo)
    np.testing.assert_allclose(x_new.detach(), x_orig.detach(), atol=1e-4)
