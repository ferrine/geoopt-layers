import geoopt
import geoopt_layers
import torch
import pytest


@pytest.fixture(autouse=True, params=[42, 41])
def seed(request):
    torch.manual_seed(request.param)


def test_distance2centroids():
    man = geoopt.Sphere()
    layer = geoopt_layers.Distance2Centroids(man, 9, 256)
    points = man.random(10, 9)
    distances = layer(points)
    assert distances.shape == (10, 256)


def test_distance_pairwise():
    man = geoopt.Sphere()
    layer = geoopt_layers.PairwiseDistances(dim=0)
    points = man.random(10, 9)
    distances = layer(points)
    assert distances.shape == (10, 10)

    layer = geoopt_layers.PairwiseDistances(dim=-2)
    points = man.random(10, 3, 9)
    distances = layer(points)
    assert distances.shape == (10, 3, 3)


def test_distance_pairwise_paired():
    man = geoopt.Sphere()
    layer = geoopt_layers.PairwiseDistances(dim=0)
    points = man.random(10, 9)
    points1 = man.random(7, 9)
    distances = layer(points, points1)
    assert distances.shape == (10, 7)

    layer = geoopt_layers.PairwiseDistances(dim=-3)
    points = man.random(10, 3, 9)
    points1 = man.random(7, 3, 9)
    distances = layer(points, points1)
    assert distances.shape == (10, 7, 3)


def test_distance_pairwise_auto():
    man = geoopt.Sphere()
    layer = geoopt_layers.PairwiseDistances(dim=0, manifold=man)
    points = man.random(10, 9)
    distances = layer(points)
    assert distances.shape == (10, 10)

    layer = geoopt_layers.PairwiseDistances(dim=-2, manifold=man)
    points = man.random(10, 3, 9)
    distances = layer(points.detach())
    assert distances.shape == (10, 3, 3)
