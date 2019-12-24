import itertools
import torch
import pytest
import geoopt
from geoopt_layers.poincare.graph import HyperbolicSplineConv


@pytest.mark.parametrize(
    "bias,sizes,degree,kernel_size,root_weight,dim,local",
    itertools.product(
        [True, False],  # bias
        [(5, 5), (5, 7)],  # sizes
        [1, 2],  # degree
        [1, 2],  # kernel_size
        [True, False],  # root_weight
        [1, 2],  # dim
        [True, False],  # local
    ),
)
def test_spline_conv(bias, sizes, kernel_size, degree, root_weight, dim, local):
    ball = geoopt.PoincareBallExact()
    ball_out = geoopt.PoincareBallExact(c=0.1)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    x = ball.random(4, 5)
    pseudo = torch.rand(edge_index.size(1), dim)
    should_pass = not (sizes[0] != sizes[1] and local and not root_weight)
    if should_pass:
        out = HyperbolicSplineConv(
            *sizes,
            bias=bias,
            ball=ball,
            ball_out=ball_out,
            kernel_size=kernel_size,
            degree=degree,
            root_weight=root_weight,
            dim=dim,
            local=local,
        )(x, edge_index, pseudo=pseudo)
        assert out.shape == (4, sizes[-1])
        ball_out.assert_check_point_on_manifold(out)
    else:
        with pytest.raises(TypeError) as e:
            HyperbolicSplineConv(
                *sizes,
                bias=bias,
                ball=ball,
                ball_out=ball_out,
                kernel_size=kernel_size,
                degree=degree,
                root_weight=root_weight,
                dim=dim,
                local=local,
            )
        assert e.match("Root should be specified")
