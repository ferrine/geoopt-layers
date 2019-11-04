import geoopt
import geoopt_layers


def test_linear_same_ball():
    ball = geoopt.PoincareBall()
    point = ball.random(2, 3, 5)
    layer = geoopt_layers.hyperbolic.MobiusLinear(5, 5, ball=ball)
    out = layer(point)
    ball.assert_attached(out)
    ball.assert_check_point_on_manifold(out)


def test_linear_new_ball():
    ball = geoopt.PoincareBall()
    ball2 = geoopt.PoincareBall()
    point = ball.random(2, 3, 5)
    layer = geoopt_layers.hyperbolic.MobiusLinear(5, 5, ball=ball, ball_out=ball2)
    out = layer(point)
    ball2.assert_attached(out)
    ball2.assert_check_point_on_manifold(out)


def test_linear_new_ball1():
    ball = geoopt.PoincareBall()
    point = ball.random(2, 3, 5)
    layer = geoopt_layers.hyperbolic.MobiusLinear(5, 5, ball=ball, ball_out=ball)
    out = layer(point)
    ball.assert_attached(out)
    ball.assert_check_point_on_manifold(out)
