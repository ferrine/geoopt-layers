import geoopt
import geoopt_layers
import numpy as np

import geoopt_layers.shape


def test_gromov_product():
    man = geoopt.Euclidean(1)
    layer = geoopt_layers.GromovProduct(man)
    x = man.random(10, 9)
    y = man.random(10, 9)
    reference = man.origin(10, 9)
    inner = layer(reference, x, y)
    assert inner.shape == (10,)
    inner_reference = (x * y).sum(-1)
    np.testing.assert_allclose(inner, inner_reference, atol=1e-5)


def test_distance2centroids():
    man = geoopt.Sphere()
    layer = geoopt_layers.Distance2Centroids(man, 9, 256)
    points = man.random(10, 9)
    distances = layer(points)
    assert distances.shape == (10, 256)


def test_distance_pairwise():
    man = geoopt.Sphere()
    layer = geoopt_layers.PairwiseDistances(manifold=man, dim=0)
    points = man.random(10, 9)
    distances = layer(points)
    assert distances.shape == (10, 10)

    layer = geoopt_layers.PairwiseDistances(manifold=man, dim=-2)
    points = man.random(10, 3, 9)
    distances = layer(points)
    assert distances.shape == (10, 3, 3)


def test_distance_pairwise_paired():
    man = geoopt.Sphere()
    layer = geoopt_layers.PairwiseDistances(manifold=man, dim=0)
    points = man.random(10, 9)
    points1 = man.random(7, 9)
    distances = layer(points, points1)
    assert distances.shape == (10, 7)

    layer = geoopt_layers.PairwiseDistances(manifold=man, dim=-3)
    points = man.random(10, 3, 9)
    points1 = man.random(7, 3, 9)
    distances = layer(points, points1)
    assert distances.shape == (10, 7, 3)


def test_distance_pairwise_auto():
    man = geoopt.Sphere()
    layer = geoopt_layers.PairwiseDistances(manifold=man, dim=0)
    points = man.random(10, 9)
    distances = layer(points)
    assert distances.shape == (10, 10)

    layer = geoopt_layers.PairwiseDistances(manifold=man, dim=-2)
    points = man.random(10, 3, 9)
    distances = layer(points.detach())
    assert distances.shape == (10, 3, 3)


def test_knn_index():
    man = geoopt.Sphere()
    layer = geoopt_layers.KNNIndex(manifold=man, dim=0, k=5)
    points = man.random(10, 9)
    idx = layer(points)
    assert idx.shape == (10, 5)


def test_knn():
    man = geoopt.Sphere()
    layer = geoopt_layers.KNN(manifold=man, dim=0, k=5)
    points = man.random(10, 9)
    knn_points = layer(points)
    assert knn_points.shape == (10, 5, 9)
    man.assert_check_point_on_manifold(knn_points)


def test_knn_permutations():
    man = geoopt.Sphere()
    layer = geoopt_layers.KNN(manifold=man, dim=-2, k=7)
    points = man.random(1, 2, 3, 9, 5)
    knn_points_1 = layer(points)
    assert knn_points_1.shape == (1, 2, 3, 9, 7, 5)

    perm1 = geoopt_layers.shape.Permute(3, 0, 1, 2, 4, contiguous=True)
    layer = geoopt_layers.KNN(manifold=man, dim=0, k=7)
    knn_points_2 = layer(perm1(points))
    assert knn_points_2.shape == (9, 7, 1, 2, 3, 5)
    knn_points_2 = knn_points_2.permute(2, 3, 4, 0, 1, 5)
    man.assert_check_point_on_manifold(knn_points_1)
    np.testing.assert_allclose(knn_points_1, knn_points_2)


def test_knn_unroll():
    man = geoopt.Sphere()
    layer = geoopt_layers.KNN(manifold=man, dim=1, k=5)
    layer_u = geoopt_layers.KNN(manifold=man, dim=1, k=5, unroll=0)
    points = man.random(7, 10, 9)
    knn_points1 = layer(points)
    knn_points2 = layer_u(points)
    assert knn_points1.shape == (7, 10, 5, 9)
    assert knn_points1.shape == knn_points2.shape
    np.testing.assert_allclose(knn_points1, knn_points2)
