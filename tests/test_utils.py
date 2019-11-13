import geoopt_layers
import torch


def test_permute():
    ten = torch.randn(1, 2, 3, 4)
    perm = geoopt_layers.Permute(0, 2, 1, 3)
    ten1 = perm(ten)

    invperm = perm.inverse()
    ten2 = invperm(ten1)
    assert ten1.shape == (1, 3, 2, 4)
    assert ten2.shape == (1, 2, 3, 4)
