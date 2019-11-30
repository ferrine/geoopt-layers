import geoopt
import geoopt_layers
import torch
import pytest


@pytest.fixture(autouse=True, params=[42, 41])
def seed(request):
    torch.manual_seed(request.param)


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


def test_linear_no_dim_change():
    ball = geoopt.PoincareBall()
    with pytest.raises(ValueError):
        geoopt_layers.poincare.MobiusLinear(5, 3, ball=ball, ball_out=ball)


def test_linear_new_ball_origin():
    ball = geoopt.PoincareBall()
    ball2 = geoopt.PoincareBall()
    point = ball.random(2, 3, 5)
    layer = geoopt_layers.poincare.MobiusLinear(
        5, 5, ball=ball, ball_out=ball2, learn_origin=True
    )
    out = layer(point)

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


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no-bias"])
def test_batch_norm(bias):
    ball = geoopt.PoincareBall()
    point = ball.random(2, 5)
    layer = geoopt_layers.poincare.MobiusBatchNorm(5, ball=ball, bias=bias)
    out = layer(point)
    assert out.shape == (2, 5)

    ball.assert_check_point_on_manifold(out)


def test_batch_norm_multi():
    ball = geoopt.PoincareBall()
    point = ball.random(2, 3, 4, 5)
    layer = geoopt_layers.poincare.MobiusBatchNorm((3, 4, 5), ball=ball)
    out = layer(point)
    assert out.shape == (2, 3, 4, 5)

    ball.assert_check_point_on_manifold(out)


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no-bias"])
def test_batch_norm_1d(bias):
    ball = geoopt.PoincareBall()
    point = ball.random(2, 5, 5, 2).permute(0, 1, 3, 2)
    layer = geoopt_layers.poincare.MobiusBatchNorm1d(2, ball=ball, bias=bias)
    out = layer(point)
    assert out.shape == (2, 5, 2, 5)

    ball.assert_check_point_on_manifold(out.permute(0, 1, 3, 2))


def test_batch_norm_1d_multi():
    ball = geoopt.PoincareBall()
    point = ball.random(2, 5, 3, 5, 2).permute(0, 1, 2, 4, 3)
    layer = geoopt_layers.poincare.MobiusBatchNorm1d((3, 2), ball=ball)
    out = layer(point)
    assert out.shape == (2, 5, 3, 2, 5)

    ball.assert_check_point_on_manifold(out.permute(0, 1, 2, 4, 3))


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no-bias"])
def test_batch_norm_2d(bias):
    ball = geoopt.PoincareBall()
    point = ball.random(2, 5, 5, 2).permute(0, 3, 1, 2)
    layer = geoopt_layers.poincare.MobiusBatchNorm2d(2, ball=ball, bias=bias)
    out = layer(point)
    assert out.shape == (2, 2, 5, 5)

    ball.assert_check_point_on_manifold(out.permute(0, 2, 3, 1))


def test_batch_norm_2d_multi():
    ball = geoopt.PoincareBall()
    point = ball.random(2, 3, 5, 5, 2).permute(0, 1, 4, 2, 3)
    layer = geoopt_layers.poincare.MobiusBatchNorm2d((3, 2), ball=ball)
    out = layer(point)
    assert out.shape == (2, 3, 2, 5, 5)

    ball.assert_check_point_on_manifold(out.permute(0, 1, 3, 4, 2))


def test_radial():
    ball = geoopt.PoincareBall()
    point = ball.random(2, 5, 5, 2).permute(0, 3, 1, 2)
    layer = geoopt_layers.poincare.RadialNd(torch.nn.ELU(), ball=ball)
    out = layer(point)
    ball.assert_check_point_on_manifold(out.permute(0, 2, 3, 1))


@pytest.mark.parametrize("squared", [True, False], ids=["squared", "not-squared"])
def test_dist_centroids_2d(squared):
    ball = geoopt.PoincareBall()
    point = ball.random(2, 5, 5, 2).permute(0, 3, 1, 2)
    layer = geoopt_layers.poincare.Distance2PoincareCentroids2d(
        centroid_shape=2, num_centroids=10, ball=ball, squared=squared
    )
    out = layer(point)
    assert not torch.isnan(out).any()
    assert out.shape == (2, 10, 5, 5)


@pytest.mark.parametrize("squared", [True, False], ids=["squared", "not-squared"])
def test_dist_centroids(squared):
    ball = geoopt.PoincareBall()
    point = ball.random(2, 5, 5, 2)
    layer = geoopt_layers.poincare.Distance2PoincareCentroids(
        centroid_shape=2, num_centroids=10, ball=ball, squared=squared
    )
    out = layer(point)
    assert not torch.isnan(out).any()
    assert out.shape == (2, 5, 5, 10)


@pytest.mark.parametrize("squared", [True, False], ids=["squared", "not-squared"])
def test_dist_centroids_2d_multi(squared):
    ball = geoopt.PoincareBall()
    point = ball.random(2, 3, 5, 5, 2).permute(0, 1, 4, 2, 3)
    layer = geoopt_layers.poincare.Distance2PoincareCentroids2d(
        centroid_shape=2, num_centroids=10, ball=ball, squared=squared
    )
    out = layer(point)
    assert not torch.isnan(out).any()
    assert out.shape == (2, 3, 10, 5, 5)


@pytest.mark.parametrize("method", ["einstein", "tangent"])
def test_weighted_centroids(method):
    ball = geoopt.PoincareBall()
    weights = torch.randn(2, 5, 17)
    layer = geoopt_layers.poincare.WeightedPoincareCentroids(
        7, 17, method=method, ball=ball
    )
    out = layer(weights)
    assert out.shape == (2, 5, 7)
    ball.assert_check_point_on_manifold(out)


@pytest.mark.parametrize("method", ["tangent"])
def test_weighted_centroids_zeros(method):
    ball = geoopt.PoincareBall()
    weights = torch.zeros(2, 5, 17)
    layer = geoopt_layers.poincare.WeightedPoincareCentroids(
        7, 17, method=method, ball=ball
    )
    out = layer(weights)
    assert out.shape == (2, 5, 7)
    ball.assert_check_point_on_manifold(out)


@pytest.mark.parametrize("method", ["einstein", "tangent"])
def test_weighted_centroids_2d(method):
    ball = geoopt.PoincareBall()
    point = ball.random(2, 5, 5, 7).permute(0, 3, 1, 2)
    layer = geoopt_layers.poincare.WeightedPoincareCentroids2d(
        2, 7, method=method, ball=ball
    )
    out = layer(point)
    assert out.shape == (2, 2, 5, 5)

    ball.assert_check_point_on_manifold(out.permute(0, 2, 3, 1))


@pytest.mark.parametrize("method", ["einstein", "tangent"])
def test_weighted_centroids_2d_multi(method):
    ball = geoopt.PoincareBall()
    point = ball.random(2, 3, 5, 5, 7).permute(0, 1, 4, 2, 3)
    layer = geoopt_layers.poincare.WeightedPoincareCentroids2d(
        2, 7, method=method, ball=ball
    )
    out = layer(point)
    assert out.shape == (2, 3, 2, 5, 5)

    ball.assert_check_point_on_manifold(out.permute(0, 1, 3, 4, 2))
