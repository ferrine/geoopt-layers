from .math import apply_radial
from geoopt_layers.base import ManifoldModule

__all__ = ["Radial", "Radial2d", "RadialNd"]


class Radial(ManifoldModule):
    """
    Apply radial nonlinearity in poincare ball.

    Point-wise nonlinearities male little sense in the Poincare ball, this function should help to
    apply any kind of radial functions.

    Parameters
    ----------
    fn : callable
        function to apply
    dim : int
        dimension to treat as manifold dimension
    ball : geoopt.PoincareBall
        manifold
    norm : bool
        do we need to norm basis vector?
    """

    def __init__(self, fn, dim=-1, *, ball, norm=True):
        super().__init__()
        self.fn = fn
        self.dim = dim
        self.ball = ball
        self.norm = norm

    def forward(self, input, basis=None):
        input = self.ball.logmap0(input, dim=self.dim)
        if basis is None:
            basis = input
        result = apply_radial(self.fn, input, basis, dim=self.dim, norm=self.norm)
        return self.ball.expmap0(result, dim=self.dim)


class RadialNd(Radial):
    def __init__(self, fn, *, ball, norm=True):
        super().__init__(fn, dim=1, ball=ball, norm=norm)


Radial2d = RadialNd
