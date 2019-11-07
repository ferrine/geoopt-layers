import geoopt_layers
import geoopt
import torch


def test_permute():
    ten = torch.randn(1, 2, 3, 4)
    man = geoopt.SphereExact()
    perm = geoopt_layers.Permute(0, 2, 1, 3, manifold=man)
    ten1 = perm(ten)
    man.assert_attached(ten1)
    invperm = perm.inverse()
    ten2 = invperm(ten1)
    assert ten1.shape == (1, 3, 2, 4)
    assert ten2.shape == (1, 2, 3, 4)
