import geoopt
from geoopt_layers.base import ManifoldModule
from .math import poincare_mean
import torch


__all__ = [
    "Distance2PoincareCentroids",
    "Distance2PoincareCentroids1d",
    "Distance2PoincareCentroids2d",
    "Distance2PoincareCentroids3d",
    "WeightedPoincareCentroids",
    "WeightedPoincareCentroids1d",
    "WeightedPoincareCentroids2d",
    "WeightedPoincareCentroids3d",
]


class Distance2PoincareCentroids(ManifoldModule):
    n = 0

    def __init__(
        self, centroid_shape: int, num_centroids: int, squared=False, *, ball,
    ):
        super().__init__()
        if not isinstance(num_centroids, int) or num_centroids < 1:
            raise TypeError("num_centroids should be int > 0")
        self.centroid_shape = centroid_shape = geoopt.utils.size2shape(centroid_shape)
        self.num_centroids = num_centroids
        self.manifold = ball
        self.log_centroids = torch.nn.Parameter(
            torch.empty((num_centroids,) + centroid_shape), requires_grad=True
        )
        self.squared = squared
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        self.log_centroids.normal_(std=self.log_centroids.shape[-1] ** -0.5)

    @property
    def centroids(self):
        return self.manifold.expmap0(self.log_centroids)

    def forward(self, input):
        input = input.unsqueeze(-self.n - 1)
        centroids = self.centroids.permute(1, 0)
        centroids = centroids.view(centroids.shape + (1,) * self.n)
        if self.squared:
            dist = self.manifold.dist2(input, centroids, dim=-self.n - 2)
        else:
            dist = self.manifold.dist(input, centroids, dim=-self.n - 2)
        return dist


class Distance2PoincareCentroids1d(Distance2PoincareCentroids):
    n = 1


class Distance2PoincareCentroids2d(Distance2PoincareCentroids):
    n = 2


class Distance2PoincareCentroids3d(Distance2PoincareCentroids):
    n = 3


class WeightedPoincareCentroids(ManifoldModule):
    n = 0

    def __init__(
        self,
        centroid_shape: int,
        num_centroids: int,
        method: str = "einstein",
        *,
        ball,
        learn_origin=True,
        std=1.0,
        lincomb=True,
    ):
        super().__init__()

        if not isinstance(num_centroids, int) or num_centroids < 1:
            raise TypeError("num_centroids should be int > 0")
        self.centroid_shape = centroid_shape = geoopt.utils.size2shape(centroid_shape)
        self.std = std
        self.num_centroids = num_centroids
        self.manifold = ball
        self.lincomb = lincomb
        self.method = method
        if lincomb:
            # to avoid scaling ambiguity, normalize basis
            self.basis_manifold = geoopt.Sphere()
        else:
            # input is normalized, this will allow to have proper output support
            self.basis_manifold = geoopt.Euclidean(ndim=1)
        self.log_centroids = geoopt.ManifoldParameter(
            torch.empty((num_centroids,) + centroid_shape), manifold=self.basis_manifold
        )
        self.learn_origin = learn_origin and method == "tangent"
        if self.learn_origin:
            self.log_origin = torch.nn.Parameter(
                torch.empty(centroid_shape), requires_grad=True
            )
        else:
            self.register_parameter("log_origin", None)
        self.reset_parameters()

    @property
    def centroids(self):
        return self.manifold.expmap0(self.log_centroids)

    @property
    def origin(self):
        if self.log_origin is not None:
            return self.manifold.expmap0(self.log_origin)
        else:
            return None

    @torch.no_grad()
    def reset_parameters(self):
        self.log_centroids.normal_(std=self.log_centroids.shape[-1] ** -0.5)
        self.log_centroids.proj_()
        if self.log_origin is not None:
            self.log_origin.zero_()

    def forward(self, weights):
        if self.origin is not None:
            origin = self.origin.view(self.origin.shape + (1,) * self.n)
        else:
            origin = None
        return poincare_mean(
            self.centroids.view(self.centroids.shape + (1,) * self.n),
            weights=weights,
            reducedim=[-self.n - 2],
            dim=-self.n - 1,
            ball=self.manifold,
            keepdim=False,
            method=self.method,
            origin=origin,
            lincomb=self.lincomb,
        )

    def extra_repr(self) -> str:
        return (
            "manifold={self.manifold}, "
            "centroid_shape={centroid_shape}, "
            "num_centroids={num_centroids}, "
            "method={method}, "
            "learn_origin={learn_origin}, "
            "lincomb={lincomb}".format(**self.__dict__, self=self)
        )


class WeightedPoincareCentroids1d(WeightedPoincareCentroids):
    n = 1


class WeightedPoincareCentroids2d(WeightedPoincareCentroids):
    n = 2


class WeightedPoincareCentroids3d(WeightedPoincareCentroids):
    n = 3
