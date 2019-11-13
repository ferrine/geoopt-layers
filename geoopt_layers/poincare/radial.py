from .math import apply_radial
from ..utils import ManifoldModule

__all__ = ["Radial", "Radial2d", "RadialNd"]


class Radial(ManifoldModule):
    def __init__(self, fn, dim=-1):
        super().__init__()
        self.fn = fn
        self.dim = dim

    def forward(self, input, basis=None):
        if basis is None:
            basis = input
        return apply_radial(self.fn, input, basis, dim=self.dim)


class RadialNd(Radial):
    def __init__(self, fn):
        super().__init__(fn, dim=1)


Radial2d = RadialNd
