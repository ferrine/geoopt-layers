import geoopt
import torch.nn
import numpy as np
import geoopt_layers
import pytest


def test_lambda_one_origin():
    sphere = geoopt.Sphere()
    point = sphere.random(1, 10)
    func = torch.nn.Linear(10, 10)
    with pytest.raises(ValueError):
        geoopt_layers.TangentLambda(func, manifold=sphere)
    layer = geoopt_layers.TangentLambda(func, manifold=sphere, origin_shape=10)
    out = layer(point)

    sphere.assert_check_point_on_manifold(out)


def test_lambda_two_origins():
    sphere = geoopt.Sphere()
    point = sphere.random(1, 10)
    func = torch.nn.Linear(10, 10)
    layer = geoopt_layers.TangentLambda(
        func, manifold=sphere, origin_shape=10, same_origin=False
    )
    out = layer(point)

    sphere.assert_check_point_on_manifold(out)


def test_lambda_no_dim_change_origins():
    sphere = geoopt.Sphere()
    func = torch.nn.Linear(10, 11)
    with pytest.raises(ValueError):
        geoopt_layers.TangentLambda(
            func,
            manifold=sphere,
            origin=sphere.origin(10),
            out_origin=sphere.origin(11),
        )


def test_remap_request_shapes():
    sphere = geoopt.Sphere()
    poincare = geoopt.PoincareBall()
    func = torch.nn.Linear(10, 13)
    with pytest.raises(ValueError):
        geoopt_layers.RemapLambda(
            func,
            source_manifold=sphere,
            target_manifold=poincare,
            source_origin_shape=None,
            target_origin_shape=10,
        )
    with pytest.raises(ValueError):
        geoopt_layers.RemapLambda(
            func,
            source_manifold=sphere,
            target_manifold=poincare,
            source_origin_shape=10,
            target_origin_shape=None,
        )


def test_remap():
    sphere = geoopt.Sphere()
    poincare = geoopt.PoincareBall()
    point = sphere.random(1, 10)
    func = torch.nn.Linear(10, 13)
    layer = geoopt_layers.RemapLambda(
        func,
        source_manifold=sphere,
        target_manifold=poincare,
        source_origin_shape=10,
        target_origin_shape=13,
    )
    out = layer(point)

    poincare.assert_check_point_on_manifold(out)


def test_remap_provided_origin():
    sphere = geoopt.Sphere()
    poincare = geoopt.PoincareBall()
    point = sphere.random(1, 10)
    func = torch.nn.Linear(10, 13)
    layer = geoopt_layers.RemapLambda(
        func,
        source_manifold=sphere,
        target_manifold=poincare,
        source_origin=sphere.origin(10),
        target_origin=poincare.origin(13),
    )
    out = layer(point)

    poincare.assert_check_point_on_manifold(out)


def test_remap_init():
    sphere = geoopt.Sphere()
    layer = geoopt_layers.Remap(source_manifold=sphere, source_origin=sphere.origin(10))
    assert layer.target_manifold is sphere


def test_expmap():
    sphere = geoopt.Sphere()
    layer = geoopt_layers.Expmap(manifold=sphere, origin=sphere.origin(10))
    smth = torch.randn(12, 3, 10)
    out = layer(smth)

    sphere.assert_check_point_on_manifold(out)


def test_logmap():
    sphere = geoopt.Sphere()
    layer = geoopt_layers.Logmap(manifold=sphere, origin=sphere.origin(10))
    out = layer(sphere.random(2, 30, 10))
    assert isinstance(out, torch.Tensor)
    assert not torch.isnan(out).any()


def test_expmap2d():
    ball = geoopt.PoincareBall()
    point = torch.randn(2, 2, 5, 5)
    layer = geoopt_layers.Expmap2d(ball, origin_shape=(2,))
    layer1 = geoopt_layers.Logmap2d(ball, origin_shape=(2,))
    out = layer(point)
    assert out.shape == (2, 2, 5, 5)

    ball.assert_check_point_on_manifold(out.permute(0, 2, 3, 1))
    reverse = layer1(out)
    np.testing.assert_allclose(point.detach(), reverse.detach(), atol=1e-4)
