import geoopt
import geoopt_layers
import pytest


def test_linear_same_ball():
    ball = geoopt.PoincareBall()
    point = ball.random(2, 3, 5)
    layer = geoopt_layers.poincare.MobiusLinear(5, 5, ball=ball)
    out = layer(point)
    ball.assert_attached(out)
    ball.assert_check_point_on_manifold(out)


def test_linear_new_ball():
    ball = geoopt.PoincareBall()
    ball2 = geoopt.PoincareBall()
    point = ball.random(2, 3, 5)
    layer = geoopt_layers.poincare.MobiusLinear(5, 5, ball=ball, ball_out=ball2)
    out = layer(point)
    ball2.assert_attached(out)
    ball2.assert_check_point_on_manifold(out)


def test_linear_new_ball1():
    ball = geoopt.PoincareBall()
    point = ball.random(2, 3, 5)
    layer = geoopt_layers.poincare.MobiusLinear(5, 5, ball=ball, ball_out=ball)
    out = layer(point)
    ball.assert_attached(out)
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
    ball2.assert_attached(out)
    ball2.assert_check_point_on_manifold(out)


def test_linear_new_ball1_origin():
    ball = geoopt.PoincareBall()
    point = ball.random(2, 3, 5)
    layer = geoopt_layers.poincare.MobiusLinear(
        5, 5, ball=ball, ball_out=ball, learn_origin=True
    )
    out = layer(point)
    ball.assert_attached(out)
    ball.assert_check_point_on_manifold(out)


def test_avg_pool():
    ball = geoopt.PoincareBall()
    point = ball.random(2, 5, 5, 2).permute(0, 3, 1, 2)
    point = ball.attach(point)
    layer = geoopt_layers.poincare.MobiusAvgPool2d((3, 3), ball=ball)
    out = layer(point)
    assert out.shape == (2, 2, 3, 3)
    ball.assert_attached(out)
    ball.assert_check_point_on_manifold(out.permute(0, 2, 3, 1))


def test_max_pool():
    ball = geoopt.PoincareBall()
    point = ball.random(2, 5, 5, 2).permute(0, 3, 1, 2)
    point = ball.attach(point)
    layer = geoopt_layers.poincare.MobiusMaxPool2d((3, 3), stride=1, ball=ball)
    out = layer(point)
    assert out.shape == (2, 2, 3, 3)
    ball.assert_attached(out)
    ball.assert_check_point_on_manifold(out.permute(0, 2, 3, 1))


def test_adaptive_max_pool():
    ball = geoopt.PoincareBall()
    point = ball.random(2, 5, 5, 2).permute(0, 3, 1, 2)
    point = ball.attach(point)
    layer = geoopt_layers.poincare.MobiusAdaptiveMaxPool2d((2, 2), ball=ball)
    out = layer(point)
    assert out.shape == (2, 2, 2, 2)
    ball.assert_attached(out)
    ball.assert_check_point_on_manifold(out.permute(0, 2, 3, 1))


def test_adaptive_avg_pool():
    ball = geoopt.PoincareBall()
    point = ball.random(2, 5, 5, 2).permute(0, 3, 1, 2)
    point = ball.attach(point)
    layer = geoopt_layers.poincare.MobiusAdaptiveAvgPool2d((2, 2), ball=ball)
    out = layer(point)
    assert out.shape == (2, 2, 2, 2)
    ball.assert_attached(out)
    ball.assert_check_point_on_manifold(out.permute(0, 2, 3, 1))
