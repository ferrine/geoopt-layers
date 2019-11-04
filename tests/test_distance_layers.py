import geoopt
import geoopt_layers


def test_distance2centroids():
    man = geoopt.Sphere()
    layer = geoopt_layers.Distance2Centroids(man, 9, 256)
    points = man.random(10, 9)
    distances = layer(points)
    assert distances.shape == (10, 256)
