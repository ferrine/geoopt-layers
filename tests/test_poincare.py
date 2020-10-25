import geoopt_layers
import itertools
import torch
import pytest
import geoopt
import numpy as np


def test_linear_same_ball(ball):
    point = ball.random(2, 3, 5)
    layer = geoopt_layers.poincare.MobiusLinear(5, 5, ball=ball)
    out = layer(point)

    ball.assert_check_point_on_manifold(out)


def test_linear_new_ball(ball_1, ball_2):
    point = ball_1.random(2, 3, 5)
    layer = geoopt_layers.poincare.MobiusLinear(
        5, 7, ball=ball_1, ball_out=ball_2, num_basis=9
    )
    out = layer(point)

    ball_2.assert_check_point_on_manifold(out)


def test_tangent_linear_same_ball(ball):
    point = ball.random(2, 3, 5)
    layer = geoopt_layers.poincare.MobiusTangentLinear(5, 5, ball=ball)
    out = layer(point)

    ball.assert_check_point_on_manifold(out)


def test_tangent_linear_new_ball(ball_1, ball_2):
    point = ball_1.random(2, 3, 5)
    layer = geoopt_layers.poincare.MobiusTangentLinear(
        5, 5, ball=ball_1, ball_out=ball_2
    )
    out = layer(point)

    ball_2.assert_check_point_on_manifold(out)


def test_tangent_linear_new_ball1(ball):
    point = ball.random(2, 3, 5)
    layer = geoopt_layers.poincare.MobiusTangentLinear(5, 5, ball=ball, ball_out=ball)
    out = layer(point)

    ball.assert_check_point_on_manifold(out)


def test_tangent_linear_dim_change(ball):
    point = ball.random(2, 3, 5)
    layer = geoopt_layers.poincare.MobiusTangentLinear(5, 3, ball=ball, ball_out=ball)
    out = layer(point)

    ball.assert_check_point_on_manifold(out)


def test_tangent_linear_new_ball_origin(ball_1, ball_2):
    point = ball_1.random(2, 3, 5)
    layer = geoopt_layers.poincare.MobiusTangentLinear(
        5, 5, ball=ball_1, ball_out=ball_2, learn_origin=True
    )
    out = layer(point)

    ball_2.assert_check_point_on_manifold(out)


def test_tangent_linear_new_ball0_origin(ball_1, ball_2):
    point = ball_1.random(2, 3, 5)
    layer = geoopt_layers.poincare.MobiusTangentLinear(
        5, 7, ball=ball_1, ball_out=ball_2, learn_origin=True
    )
    out = layer(point)
    assert out.shape == (2, 3, 7)
    ball_2.assert_check_point_on_manifold(out)


def test_tangent_linear_new_ball1_origin(ball):
    point = ball.random(2, 3, 5)
    layer = geoopt_layers.poincare.MobiusTangentLinear(
        5, 5, ball=ball, ball_out=ball, learn_origin=True
    )
    out = layer(point)

    ball.assert_check_point_on_manifold(out)


@pytest.mark.xfail(raises=NotImplementedError)
def test_avg_pool(ball):
    point = ball.random(2, 5, 5, 2).permute(0, 3, 1, 2)
    layer = geoopt_layers.poincare.MobiusAvgPool2d((3, 3), ball=ball)
    out = layer(point)
    assert out.shape == (2, 2, 3, 3)

    ball.assert_check_point_on_manifold(out.permute(0, 2, 3, 1))


def test_max_pool(ball):
    point = ball.random(2, 5, 5, 2).permute(0, 3, 1, 2)
    layer = geoopt_layers.poincare.MobiusMaxPool2d((3, 3), stride=1, ball=ball)
    out = layer(point)
    assert out.shape == (2, 2, 3, 3)

    ball.assert_check_point_on_manifold(out.permute(0, 2, 3, 1))


def test_adaptive_max_pool(ball):
    point = ball.random(2, 5, 5, 2).permute(0, 3, 1, 2)
    layer = geoopt_layers.poincare.MobiusAdaptiveMaxPool2d((2, 2), ball=ball)
    out = layer(point)
    assert out.shape == (2, 2, 2, 2)

    ball.assert_check_point_on_manifold(out.permute(0, 2, 3, 1))


@pytest.mark.xfail(raises=NotImplementedError)
def test_adaptive_avg_pool(ball):
    point = ball.random(2, 5, 5, 2).permute(0, 3, 1, 2)
    layer = geoopt_layers.poincare.MobiusAdaptiveAvgPool2d((2, 2), ball=ball)
    out = layer(point)
    assert out.shape == (2, 2, 2, 2)

    ball.assert_check_point_on_manifold(out.permute(0, 2, 3, 1))


@pytest.mark.parametrize("bias,train", itertools.product([True, False], [True, False]))
def test_batch_norm(bias, train, ball):

    point = ball.random(2, 5)
    layer = geoopt_layers.poincare.MobiusBatchNorm(5, ball=ball, bias=bias)
    layer.train(train)
    out = layer(point)
    assert out.shape == (2, 5)

    ball.assert_check_point_on_manifold(out)


@pytest.mark.parametrize("bias,train", itertools.product([True, False], [True, False]))
def test_batch_norm_multi(bias, train, ball):

    point = ball.random(2, 3, 4, 5)
    layer = geoopt_layers.poincare.MobiusBatchNorm((3, 4, 5), ball=ball, bias=bias)
    layer.train(train)
    out = layer(point)
    assert out.shape == (2, 3, 4, 5)

    ball.assert_check_point_on_manifold(out)


@pytest.mark.parametrize("bias,train", itertools.product([True, False], [True, False]))
def test_batch_norm_1d(bias, train, ball):

    point = ball.random(2, 5, 5, 2).permute(0, 1, 3, 2)
    layer = geoopt_layers.poincare.MobiusBatchNorm1d(2, ball=ball, bias=bias)
    layer.train(train)
    out = layer(point)
    assert out.shape == (2, 5, 2, 5)

    ball.assert_check_point_on_manifold(out.permute(0, 1, 3, 2))


@pytest.mark.parametrize("bias,train", itertools.product([True, False], [True, False]))
def test_batch_norm_1d_multi(bias, train, ball):

    point = ball.random(2, 5, 3, 5, 2).permute(0, 1, 2, 4, 3)
    layer = geoopt_layers.poincare.MobiusBatchNorm1d((3, 2), ball=ball, bias=bias)
    layer.train(train)
    out = layer(point)
    assert out.shape == (2, 5, 3, 2, 5)

    ball.assert_check_point_on_manifold(out.permute(0, 1, 2, 4, 3))


@pytest.mark.parametrize("bias,train", itertools.product([True, False], [True, False]))
def test_batch_norm_2d(bias, train, ball):

    point = ball.random(2, 5, 5, 2).permute(0, 3, 1, 2)
    layer = geoopt_layers.poincare.MobiusBatchNorm2d(2, ball=ball, bias=bias)
    layer.train(train)
    out = layer(point)
    assert out.shape == (2, 2, 5, 5)

    ball.assert_check_point_on_manifold(out.permute(0, 2, 3, 1))


@pytest.mark.parametrize("bias,train", itertools.product([True, False], [True, False]))
def test_batch_norm_2d_multi(bias, train, ball):

    point = ball.random(2, 3, 5, 5, 2).permute(0, 1, 4, 2, 3)
    layer = geoopt_layers.poincare.MobiusBatchNorm2d((3, 2), ball=ball, bias=bias)
    layer.train(train)
    out = layer(point)
    assert out.shape == (2, 3, 2, 5, 5)

    ball.assert_check_point_on_manifold(out.permute(0, 1, 3, 4, 2))


def test_radial(ball):

    point = ball.random(2, 5, 5, 2).permute(0, 3, 1, 2)
    layer = geoopt_layers.poincare.RadialNd(torch.nn.ELU(), ball=ball)
    out = layer(point)
    ball.assert_check_point_on_manifold(out.permute(0, 2, 3, 1))


@pytest.mark.parametrize(
    "squared,train,zero", itertools.product([True, False], [True, False], [True, False])
)
def test_dist_centroids_2d(squared, train, zero, ball):

    point = ball.random(2, 5, 5, 2).permute(0, 3, 1, 2)
    layer = geoopt_layers.poincare.Distance2PoincareCentroids2d(
        centroid_shape=2, num_centroids=10, ball=ball, squared=squared
    ).train(train)
    out = layer(point)
    assert not torch.isnan(out).any()
    assert out.shape == (2, 10, 5, 5)


@pytest.mark.parametrize(
    "squared,train", itertools.product([True, False], [True, False])
)
def test_dist_centroids(squared, train, ball):

    point = ball.random(2, 5, 5, 2)
    layer = geoopt_layers.poincare.Distance2PoincareCentroids(
        centroid_shape=2, num_centroids=10, ball=ball, squared=squared
    ).train(train)
    out = layer(point)
    assert not torch.isnan(out).any()
    assert out.shape == (2, 5, 5, 10)


@pytest.mark.parametrize(
    "squared,train", itertools.product([True, False], [True, False])
)
def test_dist_centroids_1d_multi(squared, train, ball):

    point = ball.random(2, 3, 5, 5, 2).permute(0, 1, 2, 4, 3)
    layer = geoopt_layers.poincare.Distance2PoincareCentroids1d(
        centroid_shape=2, num_centroids=10, ball=ball, squared=squared
    ).train(train)
    out = layer(point)
    assert not torch.isnan(out).any()
    assert out.shape == (2, 3, 5, 10, 5)


@pytest.mark.parametrize(
    "squared,train", itertools.product([True, False], [True, False])
)
def test_dist_centroids_2d_multi(squared, train, ball):

    point = ball.random(2, 3, 5, 5, 2).permute(0, 1, 4, 2, 3)
    layer = geoopt_layers.poincare.Distance2PoincareCentroids2d(
        centroid_shape=2, num_centroids=10, ball=ball, squared=squared
    ).train(train)
    out = layer(point)
    assert not torch.isnan(out).any()
    assert out.shape == (2, 3, 10, 5, 5)


@pytest.mark.parametrize(
    "method,train", itertools.product(["einstein", "tangent"], [True, False]),
)
def test_weighted_centroids(method, train, ball):

    weights = torch.randn(2, 5, 17)
    layer = geoopt_layers.poincare.WeightedPoincareCentroids(
        7, 17, method=method, ball=ball
    ).train(train)
    out = layer(weights)
    assert out.shape == (2, 5, 7)
    ball.assert_check_point_on_manifold(out)


@pytest.mark.parametrize("method,train", itertools.product(["tangent"], [True, False]))
def test_weighted_centroids_zeros(method, train, ball):

    weights = torch.zeros(2, 5, 17)
    layer = geoopt_layers.poincare.WeightedPoincareCentroids(
        7, 17, method=method, ball=ball
    ).train(train)
    out = layer(weights)
    assert out.shape == (2, 5, 7)
    ball.assert_check_point_on_manifold(out)


@pytest.mark.parametrize(
    "method,train", itertools.product(["einstein", "tangent"], [True, False]),
)
def test_weighted_centroids_1d(method, train, ball):

    point = ball.random(2, 5, 5, 7).permute(0, 1, 3, 2)
    layer = geoopt_layers.poincare.WeightedPoincareCentroids1d(
        2, 7, method=method, ball=ball
    ).train(train)
    out = layer(point)
    assert out.shape == (2, 5, 2, 5)

    ball.assert_check_point_on_manifold(out.permute(0, 1, 3, 2))


@pytest.mark.parametrize(
    "method,train", itertools.product(["einstein", "tangent"], [True, False]),
)
def test_weighted_centroids_1d_multi(method, train, ball):

    point = ball.random(2, 3, 5, 5, 7).permute(0, 1, 2, 4, 3)
    layer = geoopt_layers.poincare.WeightedPoincareCentroids1d(
        2, 7, method=method, ball=ball
    ).train(train)
    out = layer(point)
    assert out.shape == (2, 3, 5, 2, 5)

    ball.assert_check_point_on_manifold(out.permute(0, 1, 2, 4, 3))


@pytest.mark.parametrize(
    "method,train", itertools.product(["einstein", "tangent"], [True, False]),
)
def test_weighted_centroids_2d(method, train, ball):

    point = ball.random(2, 5, 5, 7).permute(0, 3, 1, 2)
    layer = geoopt_layers.poincare.WeightedPoincareCentroids2d(
        2, 7, method=method, ball=ball
    ).train(train)
    out = layer(point)
    assert out.shape == (2, 2, 5, 5)

    ball.assert_check_point_on_manifold(out.permute(0, 2, 3, 1))


@pytest.mark.parametrize(
    "method,train", itertools.product(["einstein", "tangent"], [True, False]),
)
def test_weighted_centroids_2d_multi(method, train, ball):

    point = ball.random(2, 3, 5, 5, 7).permute(0, 1, 4, 2, 3)
    layer = geoopt_layers.poincare.WeightedPoincareCentroids2d(
        2, 7, method=method, ball=ball
    ).train(train)
    out = layer(point)
    assert out.shape == (2, 3, 2, 5, 5)

    ball.assert_check_point_on_manifold(out.permute(0, 1, 3, 4, 2))


@pytest.mark.parametrize(
    "squared,train,signed",
    itertools.product([True, False], [True, False], [True, False]),
)
def test_dist_planes_2d(squared, train, signed, ball):

    point = ball.random(2, 5, 5, 2).permute(0, 3, 1, 2)
    layer = geoopt_layers.poincare.Distance2PoincareHyperplanes2d(
        plane_shape=2, num_planes=10, ball=ball, squared=squared, signed=signed,
    ).train(train)
    out = layer(point)
    assert not torch.isnan(out).any()
    assert out.shape == (2, 10, 5, 5)


@pytest.mark.parametrize(
    "squared,train,signed",
    itertools.product([True, False], [True, False], [True, False]),
)
def test_dist_planes(squared, train, signed, ball):

    point = ball.random(2, 5, 5, 2)
    layer = geoopt_layers.poincare.Distance2PoincareHyperplanes(
        plane_shape=2, num_planes=10, ball=ball, squared=squared, signed=signed,
    ).train(train)
    out = layer(point)
    assert not torch.isnan(out).any()
    assert out.shape == (2, 5, 5, 10)


@pytest.mark.parametrize(
    "squared,train,signed",
    itertools.product([True, False], [True, False], [True, False]),
)
def test_dist_planes_1d_multi(squared, train, signed, ball):

    point = ball.random(2, 3, 5, 5, 2).permute(0, 1, 2, 4, 3)
    layer = geoopt_layers.poincare.Distance2PoincareHyperplanes1d(
        plane_shape=2, num_planes=10, ball=ball, squared=squared, signed=signed,
    ).train(train)
    out = layer(point)
    assert not torch.isnan(out).any()
    assert out.shape == (2, 3, 5, 10, 5)


@pytest.mark.parametrize(
    "squared,train,signed",
    itertools.product([True, False], [True, False], [True, False]),
)
@torch.no_grad()
def test_dist_planes_2d_multi(squared, train, signed, ball):

    point = ball.random(2, 3, 5, 5, 2).permute(0, 1, 4, 2, 3)
    layer = geoopt_layers.poincare.Distance2PoincareHyperplanes2d(
        plane_shape=2, num_planes=10, ball=ball, squared=squared, signed=signed,
    ).train(train)
    out = layer(point)
    assert not torch.isnan(out).any()
    assert out.shape == (2, 3, 10, 5, 5)


@pytest.mark.parametrize("mm", [True, False])
@pytest.mark.xfail(raises=NotImplementedError)
@torch.no_grad()
def test_average_equals_conv(mm, ball):

    conv = geoopt_layers.poincare.MobiusConv2d(5, 3, dim_out=5, ball=ball, matmul=mm)
    with torch.no_grad():
        if mm:
            torch.nn.init.eye_(conv.weight_mm)
        conv.weight_avg.fill_(1)
    points = ball.random(1, 1, 3, 3, 5)
    avg1 = geoopt_layers.poincare.math.poincare_mean(points, dim=-1, ball=ball)
    avg2 = conv(points.permute(0, 1, 4, 2, 3)).detach().view(-1)
    np.testing.assert_allclose(avg1, avg2, atol=1e-5, rtol=1e-5)
    ball.assert_check_point_on_manifold(avg2)


@pytest.mark.parametrize("mm", [True, False])
@pytest.mark.xfail(raises=NotImplementedError)
@torch.no_grad()
def test_weighted_average_equals_conv(mm, ball):

    conv = geoopt_layers.poincare.MobiusConv2d(5, 3, dim_out=5, ball=ball, matmul=mm)
    with torch.no_grad():
        if mm:
            torch.nn.init.eye_(conv.weight_mm)
        conv.weight_avg.normal_()
        weight_avg = conv.weight_avg.detach().view(3, 3)
    points = ball.random(1, 1, 3, 3, 5)
    avg1 = geoopt_layers.poincare.math.poincare_mean(
        points, weight_avg, dim=-1, ball=ball
    )
    avg2 = conv(points.permute(0, 1, 4, 2, 3)).detach().view(-1)
    np.testing.assert_allclose(avg1, avg2, atol=1e-5, rtol=1e-5)
    ball.assert_check_point_on_manifold(avg2)


@pytest.mark.parametrize("mm", [True, False])
@pytest.mark.xfail(raises=NotImplementedError)
@torch.no_grad()
def test_two_points_average_equals_conv(mm, ball):

    conv = geoopt_layers.poincare.MobiusConv2d(
        5, 3, dim_out=5, points_in=2, ball=ball, matmul=mm
    )
    with torch.no_grad():
        if mm:
            torch.nn.init.eye_(conv.weight_mm)
        conv.weight_avg.fill_(1)
    points = ball.random(1, 2, 3, 3, 5)
    avg1 = geoopt_layers.poincare.math.poincare_mean(points, dim=-1, ball=ball)
    avg2 = conv(points.permute(0, 1, 4, 2, 3)).detach().view(-1)
    np.testing.assert_allclose(avg1, avg2, atol=1e-5, rtol=1e-5)
    ball.assert_check_point_on_manifold(avg2)


@pytest.mark.parametrize("mm", [True, False])
@pytest.mark.xfail(raises=NotImplementedError)
@torch.no_grad()
def test_two_points_weighted_average_equals_conv(mm, ball):

    conv = geoopt_layers.poincare.MobiusConv2d(
        5, 3, dim_out=5, points_in=2, ball=ball, matmul=mm
    )
    with torch.no_grad():
        if mm:
            torch.nn.init.eye_(conv.weight_mm)

        conv.weight_avg.normal_()
        weight_avg = conv.weight_avg.detach().view(2, 3, 3)
    points = ball.random(1, 2, 3, 3, 5)
    avg1 = geoopt_layers.poincare.math.poincare_mean(
        points, weight_avg, dim=-1, ball=ball
    )
    avg2 = conv(points.permute(0, 1, 4, 2, 3)).detach().view(-1)
    np.testing.assert_allclose(avg1, avg2, atol=1e-5, rtol=1e-5)
    ball.assert_check_point_on_manifold(avg2)


@pytest.mark.xfail(raises=NotImplementedError)
def test_random_init_mobius_conv(ball):

    conv = geoopt_layers.poincare.MobiusConv2d(
        5, 3, dim_out=7, points_in=2, points_out=4, ball=ball
    )
    points = ball.random(3, 2, 3, 3, 5).permute(0, 1, 4, 2, 3)
    out = conv(points)
    assert out.shape == (3, 4, 7, 1, 1)
    ball.assert_check_point_on_manifold(out.permute(0, 1, 3, 4, 2))


def test_poincare_mean_scatter(ball):

    points = ball.random(10, 5, std=1 / 5 ** 0.5)
    means = geoopt_layers.poincare.math.poincare_mean_scatter(
        points, index=torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]), ball=ball
    )
    assert means.shape == (2, 5)
    mean_1 = geoopt_layers.poincare.math.poincare_mean(points[:5], ball=ball)
    mean_2 = geoopt_layers.poincare.math.poincare_mean(points[5:], ball=ball)
    np.testing.assert_allclose(means[0].detach(), mean_1.detach(), atol=1e-5)
    np.testing.assert_allclose(means[1].detach(), mean_2.detach(), atol=1e-5)


@pytest.mark.parametrize(
    "linkomb,method", itertools.product([True, False], ["einstein", "tangent"])
)
def test_poincare_mean_scatter_tangent(linkomb, method, ball):
    points = ball.random(10, 5, std=1 / 5 ** 0.5)
    means = geoopt_layers.poincare.math.poincare_mean_scatter(
        points,
        index=torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
        ball=ball,
        method=method,
        lincomb=linkomb,
    )
    assert means.shape == (2, 5)
    ball.assert_check_point_on_manifold(means)


@pytest.mark.parametrize(
    "alpha,beta,gamma",
    itertools.product(
        [100.0, 1.0, 0.05, 0.0], [100.0, 1.0, 0.05, 0.0], [0.1, 0.5, 1.0]
    ),
)
def test_noise_layer(alpha, beta, gamma, ball):

    noise = geoopt_layers.poincare.Noise(alpha, beta, gamma, ball=ball)
    input = ball.random(10, 5)
    output = noise(input)
    ball.assert_check_point_on_manifold(output)


@pytest.mark.parametrize(
    "granularity,grad", itertools.product([0.01, 1, 10], [True, False])
)
def test_noise_layer(granularity, grad, ball):

    noise = geoopt_layers.poincare.Discretization(granularity, ball=ball, grad=grad)
    input = ball.random(10, 5)
    output = noise(input)
    ball.assert_check_point_on_manifold(output)


def test_linear_layer_expanded():
    ball = geoopt.Stereographic(k=0.0)
    hyp_linear = geoopt_layers.poincare.MobiusLinear(10, 15, ball=ball, num_basis=20)
    input = torch.randn(3, 10)
    output_hyperbolic = hyp_linear(input)
    ball.assert_check_point_on_manifold(output_hyperbolic)


@pytest.mark.parametrize(
    "bias", itertools.product([True, False]),
)
def test_gromov_2d(ball, bias):

    point = ball.random(2, 5, 5, 2).permute(0, 3, 1, 2)
    layer = geoopt_layers.poincare.GromovProductHyperbolic2d(2, 10, ball=ball,)
    out = layer(point)
    assert not torch.isnan(out).any()
    assert out.shape == (2, 10, 5, 5)
    out.sum().backward()


def test_dist_gromov(ball):
    point = ball.random(2, 5, 5, 2)
    layer = geoopt_layers.poincare.GromovProductHyperbolic(2, 10, ball=ball,)
    out = layer(point)
    assert not torch.isnan(out).any()
    assert out.shape == (2, 5, 5, 10)


def test_gromov_1d_multi(ball):
    point = ball.random(2, 3, 5, 5, 2).permute(0, 1, 2, 4, 3)
    layer = geoopt_layers.poincare.GromovProductHyperbolic1d(2, 10, ball=ball,)
    out = layer(point)
    assert not torch.isnan(out).any()
    assert out.shape == (2, 3, 5, 10, 5)


def test_gromov_2d_multi(ball):
    point = ball.random(2, 3, 5, 5, 2).permute(0, 1, 4, 2, 3)
    layer = geoopt_layers.poincare.GromovProductHyperbolic2d(2, 10, ball=ball,)
    out = layer(point)
    assert not torch.isnan(out).any()
    assert out.shape == (2, 3, 10, 5, 5)
