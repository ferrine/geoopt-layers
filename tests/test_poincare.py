import geoopt
import geoopt_layers
import itertools
import torch
import pytest
import numpy as np


def test_linear_same_ball():
    ball = geoopt.PoincareBall()
    point = ball.random(2, 3, 5)
    layer = geoopt_layers.poincare.MobiusLinear(5, 5, ball=ball)
    out = layer(point)

    ball.assert_check_point_on_manifold(out)


def test_linear_new_ball():
    ball = geoopt.PoincareBall()
    ball2 = geoopt.PoincareBall()
    point = ball.random(2, 3, 5)
    layer = geoopt_layers.poincare.MobiusLinear(5, 5, ball=ball, ball_out=ball2)
    out = layer(point)

    ball2.assert_check_point_on_manifold(out)


def test_linear_new_ball1():
    ball = geoopt.PoincareBall()
    point = ball.random(2, 3, 5)
    layer = geoopt_layers.poincare.MobiusLinear(5, 5, ball=ball, ball_out=ball)
    out = layer(point)

    ball.assert_check_point_on_manifold(out)


def test_linear_dim_change():
    ball = geoopt.PoincareBall()
    point = ball.random(2, 3, 5)
    layer = geoopt_layers.poincare.MobiusLinear(5, 3, ball=ball, ball_out=ball)
    out = layer(point)

    ball.assert_check_point_on_manifold(out)


def test_linear_new_ball_origin():
    ball = geoopt.PoincareBall()
    ball2 = geoopt.PoincareBall()
    point = ball.random(2, 3, 5)
    layer = geoopt_layers.poincare.MobiusLinear(
        5, 5, ball=ball, ball_out=ball2, learn_origin=True
    )
    out = layer(point)

    ball2.assert_check_point_on_manifold(out)


def test_linear_new_ball_origin():
    ball = geoopt.PoincareBall()
    ball2 = geoopt.PoincareBall()
    point = ball.random(2, 3, 5)
    layer = geoopt_layers.poincare.MobiusLinear(
        5, 7, ball=ball, ball_out=ball2, learn_origin=True
    )
    out = layer(point)
    assert out.shape == (2, 3, 7)
    ball2.assert_check_point_on_manifold(out)


def test_linear_new_ball1_origin():
    ball = geoopt.PoincareBall()
    point = ball.random(2, 3, 5)
    layer = geoopt_layers.poincare.MobiusLinear(
        5, 5, ball=ball, ball_out=ball, learn_origin=True
    )
    out = layer(point)

    ball.assert_check_point_on_manifold(out)


def test_avg_pool():
    ball = geoopt.PoincareBall()
    point = ball.random(2, 5, 5, 2).permute(0, 3, 1, 2)
    layer = geoopt_layers.poincare.MobiusAvgPool2d((3, 3), ball=ball)
    out = layer(point)
    assert out.shape == (2, 2, 3, 3)

    ball.assert_check_point_on_manifold(out.permute(0, 2, 3, 1))


def test_max_pool():
    ball = geoopt.PoincareBall()
    point = ball.random(2, 5, 5, 2).permute(0, 3, 1, 2)
    layer = geoopt_layers.poincare.MobiusMaxPool2d((3, 3), stride=1, ball=ball)
    out = layer(point)
    assert out.shape == (2, 2, 3, 3)

    ball.assert_check_point_on_manifold(out.permute(0, 2, 3, 1))


def test_adaptive_max_pool():
    ball = geoopt.PoincareBall()
    point = ball.random(2, 5, 5, 2).permute(0, 3, 1, 2)
    layer = geoopt_layers.poincare.MobiusAdaptiveMaxPool2d((2, 2), ball=ball)
    out = layer(point)
    assert out.shape == (2, 2, 2, 2)

    ball.assert_check_point_on_manifold(out.permute(0, 2, 3, 1))


def test_adaptive_avg_pool():
    ball = geoopt.PoincareBall()
    point = ball.random(2, 5, 5, 2).permute(0, 3, 1, 2)
    layer = geoopt_layers.poincare.MobiusAdaptiveAvgPool2d((2, 2), ball=ball)
    out = layer(point)
    assert out.shape == (2, 2, 2, 2)

    ball.assert_check_point_on_manifold(out.permute(0, 2, 3, 1))


@pytest.mark.parametrize("bias,train", itertools.product([True, False], [True, False]))
def test_batch_norm(bias, train):
    ball = geoopt.PoincareBall()
    point = ball.random(2, 5)
    layer = geoopt_layers.poincare.MobiusBatchNorm(5, ball=ball, bias=bias)
    layer.train(train)
    out = layer(point)
    assert out.shape == (2, 5)

    ball.assert_check_point_on_manifold(out)


@pytest.mark.parametrize("bias,train", itertools.product([True, False], [True, False]))
def test_batch_norm_multi(bias, train):
    ball = geoopt.PoincareBall()
    point = ball.random(2, 3, 4, 5)
    layer = geoopt_layers.poincare.MobiusBatchNorm((3, 4, 5), ball=ball, bias=bias)
    layer.train(train)
    out = layer(point)
    assert out.shape == (2, 3, 4, 5)

    ball.assert_check_point_on_manifold(out)


@pytest.mark.parametrize("bias,train", itertools.product([True, False], [True, False]))
def test_batch_norm_1d(bias, train):
    ball = geoopt.PoincareBall()
    point = ball.random(2, 5, 5, 2).permute(0, 1, 3, 2)
    layer = geoopt_layers.poincare.MobiusBatchNorm1d(2, ball=ball, bias=bias)
    layer.train(train)
    out = layer(point)
    assert out.shape == (2, 5, 2, 5)

    ball.assert_check_point_on_manifold(out.permute(0, 1, 3, 2))


@pytest.mark.parametrize("bias,train", itertools.product([True, False], [True, False]))
def test_batch_norm_1d_multi(bias, train):
    ball = geoopt.PoincareBall()
    point = ball.random(2, 5, 3, 5, 2).permute(0, 1, 2, 4, 3)
    layer = geoopt_layers.poincare.MobiusBatchNorm1d((3, 2), ball=ball, bias=bias)
    layer.train(train)
    out = layer(point)
    assert out.shape == (2, 5, 3, 2, 5)

    ball.assert_check_point_on_manifold(out.permute(0, 1, 2, 4, 3))


@pytest.mark.parametrize("bias,train", itertools.product([True, False], [True, False]))
def test_batch_norm_2d(bias, train):
    ball = geoopt.PoincareBall()
    point = ball.random(2, 5, 5, 2).permute(0, 3, 1, 2)
    layer = geoopt_layers.poincare.MobiusBatchNorm2d(2, ball=ball, bias=bias)
    layer.train(train)
    out = layer(point)
    assert out.shape == (2, 2, 5, 5)

    ball.assert_check_point_on_manifold(out.permute(0, 2, 3, 1))


@pytest.mark.parametrize("bias,train", itertools.product([True, False], [True, False]))
def test_batch_norm_2d_multi(bias, train):
    ball = geoopt.PoincareBall()
    point = ball.random(2, 3, 5, 5, 2).permute(0, 1, 4, 2, 3)
    layer = geoopt_layers.poincare.MobiusBatchNorm2d((3, 2), ball=ball, bias=bias)
    layer.train(train)
    out = layer(point)
    assert out.shape == (2, 3, 2, 5, 5)

    ball.assert_check_point_on_manifold(out.permute(0, 1, 3, 4, 2))


def test_radial():
    ball = geoopt.PoincareBall()
    point = ball.random(2, 5, 5, 2).permute(0, 3, 1, 2)
    layer = geoopt_layers.poincare.RadialNd(torch.nn.ELU(), ball=ball)
    out = layer(point)
    ball.assert_check_point_on_manifold(out.permute(0, 2, 3, 1))


@pytest.mark.parametrize(
    "squared,train,zero", itertools.product([True, False], [True, False], [True, False])
)
def test_dist_centroids_2d(squared, train, zero):
    ball = geoopt.PoincareBall()
    point = ball.random(2, 5, 5, 2).permute(0, 3, 1, 2)
    layer = geoopt_layers.poincare.Distance2PoincareCentroids2d(
        centroid_shape=2, num_centroids=10, ball=ball, squared=squared, zero=zero
    ).train(train)
    out = layer(point)
    assert not torch.isnan(out).any()
    assert out.shape == (2, 10, 5, 5)


@pytest.mark.parametrize(
    "squared,train,zero", itertools.product([True, False], [True, False], [True, False])
)
def test_dist_centroids(squared, train, zero):
    ball = geoopt.PoincareBall()
    point = ball.random(2, 5, 5, 2)
    layer = geoopt_layers.poincare.Distance2PoincareCentroids(
        centroid_shape=2, num_centroids=10, ball=ball, squared=squared, zero=zero
    ).train(train)
    out = layer(point)
    assert not torch.isnan(out).any()
    assert out.shape == (2, 5, 5, 10)


@pytest.mark.parametrize(
    "squared,train,zero", itertools.product([True, False], [True, False], [True, False])
)
def test_dist_centroids_1d_multi(squared, train, zero):
    ball = geoopt.PoincareBall()
    point = ball.random(2, 3, 5, 5, 2).permute(0, 1, 2, 4, 3)
    layer = geoopt_layers.poincare.Distance2PoincareCentroids1d(
        centroid_shape=2, num_centroids=10, ball=ball, squared=squared, zero=zero
    ).train(train)
    out = layer(point)
    assert not torch.isnan(out).any()
    assert out.shape == (2, 3, 5, 10, 5)


@pytest.mark.parametrize(
    "squared,train,zero", itertools.product([True, False], [True, False], [True, False])
)
def test_dist_centroids_2d_multi(squared, train, zero):
    ball = geoopt.PoincareBall()
    point = ball.random(2, 3, 5, 5, 2).permute(0, 1, 4, 2, 3)
    layer = geoopt_layers.poincare.Distance2PoincareCentroids2d(
        centroid_shape=2, num_centroids=10, ball=ball, squared=squared, zero=zero
    ).train(train)
    out = layer(point)
    assert not torch.isnan(out).any()
    assert out.shape == (2, 3, 10, 5, 5)


@pytest.mark.parametrize(
    "method,train,zero",
    itertools.product(["einstein", "tangent"], [True, False], [True, False]),
)
def test_weighted_centroids(method, train, zero):
    ball = geoopt.PoincareBall()
    weights = torch.randn(2, 5, 17)
    layer = geoopt_layers.poincare.WeightedPoincareCentroids(
        7, 17, method=method, ball=ball, zero=zero
    ).train(train)
    out = layer(weights)
    assert out.shape == (2, 5, 7)
    ball.assert_check_point_on_manifold(out)


@pytest.mark.parametrize(
    "method,train,zero", itertools.product(["tangent"], [True, False], [True, False])
)
def test_weighted_centroids_zeros(method, train, zero):
    ball = geoopt.PoincareBall()
    weights = torch.zeros(2, 5, 17)
    layer = geoopt_layers.poincare.WeightedPoincareCentroids(
        7, 17, method=method, ball=ball, zero=zero
    ).train(train)
    out = layer(weights)
    assert out.shape == (2, 5, 7)
    ball.assert_check_point_on_manifold(out)


@pytest.mark.parametrize(
    "method,train,zero",
    itertools.product(["einstein", "tangent"], [True, False], [True, False]),
)
def test_weighted_centroids_1d(method, train, zero):
    ball = geoopt.PoincareBall()
    point = ball.random(2, 5, 5, 7).permute(0, 1, 3, 2)
    layer = geoopt_layers.poincare.WeightedPoincareCentroids1d(
        2, 7, method=method, ball=ball, zero=zero
    ).train(train)
    out = layer(point)
    assert out.shape == (2, 5, 2, 5)

    ball.assert_check_point_on_manifold(out.permute(0, 1, 3, 2))


@pytest.mark.parametrize(
    "method,train,zero",
    itertools.product(["einstein", "tangent"], [True, False], [True, False]),
)
def test_weighted_centroids_1d_multi(method, train, zero):
    ball = geoopt.PoincareBall()
    point = ball.random(2, 3, 5, 5, 7).permute(0, 1, 2, 4, 3)
    layer = geoopt_layers.poincare.WeightedPoincareCentroids1d(
        2, 7, method=method, ball=ball, zero=zero
    ).train(train)
    out = layer(point)
    assert out.shape == (2, 3, 5, 2, 5)

    ball.assert_check_point_on_manifold(out.permute(0, 1, 2, 4, 3))


@pytest.mark.parametrize(
    "method,train,zero",
    itertools.product(["einstein", "tangent"], [True, False], [True, False]),
)
def test_weighted_centroids_2d(method, train, zero):
    ball = geoopt.PoincareBall()
    point = ball.random(2, 5, 5, 7).permute(0, 3, 1, 2)
    layer = geoopt_layers.poincare.WeightedPoincareCentroids2d(
        2, 7, method=method, ball=ball, zero=zero
    ).train(train)
    out = layer(point)
    assert out.shape == (2, 2, 5, 5)

    ball.assert_check_point_on_manifold(out.permute(0, 2, 3, 1))


@pytest.mark.parametrize(
    "method,train,zero",
    itertools.product(["einstein", "tangent"], [True, False], [True, False]),
)
def test_weighted_centroids_2d_multi(method, train, zero):
    ball = geoopt.PoincareBall()
    point = ball.random(2, 3, 5, 5, 7).permute(0, 1, 4, 2, 3)
    layer = geoopt_layers.poincare.WeightedPoincareCentroids2d(
        2, 7, method=method, ball=ball, zero=zero
    ).train(train)
    out = layer(point)
    assert out.shape == (2, 3, 2, 5, 5)

    ball.assert_check_point_on_manifold(out.permute(0, 1, 3, 4, 2))


@pytest.mark.parametrize(
    "squared,train,zero,signed",
    itertools.product([True, False], [True, False], [True, False], [True, False]),
)
def test_dist_planes_2d(squared, train, zero, signed):
    ball = geoopt.PoincareBall()
    point = ball.random(2, 5, 5, 2).permute(0, 3, 1, 2)
    layer = geoopt_layers.poincare.Distance2PoincareHyperplanes2d(
        plane_shape=2,
        num_planes=10,
        ball=ball,
        squared=squared,
        zero=zero,
        signed=signed,
    ).train(train)
    out = layer(point)
    assert not torch.isnan(out).any()
    assert out.shape == (2, 10, 5, 5)


@pytest.mark.parametrize(
    "squared,train,zero,signed",
    itertools.product([True, False], [True, False], [True, False], [True, False]),
)
def test_dist_planes(squared, train, zero, signed):
    ball = geoopt.PoincareBall()
    point = ball.random(2, 5, 5, 2)
    layer = geoopt_layers.poincare.Distance2PoincareHyperplanes(
        plane_shape=2,
        num_planes=10,
        ball=ball,
        squared=squared,
        zero=zero,
        signed=signed,
    ).train(train)
    out = layer(point)
    assert not torch.isnan(out).any()
    assert out.shape == (2, 5, 5, 10)


@pytest.mark.parametrize(
    "squared,train,zero,signed",
    itertools.product([True, False], [True, False], [True, False], [True, False]),
)
def test_dist_planes_1d_multi(squared, train, zero, signed):
    ball = geoopt.PoincareBall()
    point = ball.random(2, 3, 5, 5, 2).permute(0, 1, 2, 4, 3)
    layer = geoopt_layers.poincare.Distance2PoincareHyperplanes1d(
        plane_shape=2,
        num_planes=10,
        ball=ball,
        squared=squared,
        zero=zero,
        signed=signed,
    ).train(train)
    out = layer(point)
    assert not torch.isnan(out).any()
    assert out.shape == (2, 3, 5, 10, 5)


@pytest.mark.parametrize(
    "squared,train,zero,signed",
    itertools.product([True, False], [True, False], [True, False], [True, False]),
)
def test_dist_planes_2d_multi(squared, train, zero, signed):
    ball = geoopt.PoincareBall()
    point = ball.random(2, 3, 5, 5, 2).permute(0, 1, 4, 2, 3)
    layer = geoopt_layers.poincare.Distance2PoincareHyperplanes2d(
        plane_shape=2,
        num_planes=10,
        ball=ball,
        squared=squared,
        zero=zero,
        signed=signed,
    ).train(train)
    out = layer(point)
    assert not torch.isnan(out).any()
    assert out.shape == (2, 3, 10, 5, 5)


@pytest.mark.parametrize("mm", [True, False])
def test_average_equals_conv(mm):
    ball = geoopt.PoincareBall()
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
def test_weighted_average_equals_conv(mm):
    ball = geoopt.PoincareBall()
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
def test_two_points_average_equals_conv(mm):
    ball = geoopt.PoincareBall()
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
def test_two_points_weighted_average_equals_conv(mm):
    ball = geoopt.PoincareBall()
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


def test_random_init_mobius_conv():
    ball = geoopt.PoincareBall()
    conv = geoopt_layers.poincare.MobiusConv2d(
        5, 3, dim_out=7, points_in=2, points_out=4, ball=ball
    )
    points = ball.random(3, 2, 3, 3, 5).permute(0, 1, 4, 2, 3)
    out = conv(points)
    assert out.shape == (3, 4, 7, 1, 1)
    ball.assert_check_point_on_manifold(out.permute(0, 1, 3, 4, 2))


def test_poincare_mean_scatter():
    ball = geoopt.PoincareBall()
    points = ball.random(10, 5, std=1 / 5 ** 0.5)
    means = geoopt_layers.poincare.math.poincare_mean_scatter(
        points, index=torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]), ball=ball
    )
    assert means.shape == (2, 5)
    mean_1 = geoopt_layers.poincare.math.poincare_mean(points[:5], ball=ball)
    mean_2 = geoopt_layers.poincare.math.poincare_mean(points[5:], ball=ball)
    np.testing.assert_allclose(means[0], mean_1, atol=1e-5)
    np.testing.assert_allclose(means[1], mean_2, atol=1e-5)


@pytest.mark.parametrize(
    "linkomb,method", itertools.product([True, False], ["einstein", "tangent"])
)
def test_poincare_mean_scatter_tangent(linkomb, method):
    ball = geoopt.PoincareBall()
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
    "alpha,beta",
    itertools.product(
        [100.0, 1.0, 0.05, 0.0],
        [100.0, 1.0, 0.05, 0.0],
    ),
)
def test_noise_layer(alpha, beta):
    ball = geoopt.PoincareBall()
    noise = geoopt_layers.poincare.Noise(alpha, beta, ball=ball)
    input = ball.random(10, 5)
    output = noise(input)
    ball.assert_check_point_on_manifold(output)
