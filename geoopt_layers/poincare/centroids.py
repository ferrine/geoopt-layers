import geoopt
from ..utils import ManifoldModule
from .. import distance
from .math import poincare_lincomb
import torch


class Distance2PoincareCentroids(distance.Distance2Centroids):
    def __init__(
        self, centroid_shape: int, num_centroids: int, squared=True, *, ball, std=1.0
    ):
        self.std = std
        super().__init__(
            ball,
            centroid_shape=centroid_shape,
            num_centroids=num_centroids,
            squared=squared,
        )

    @torch.no_grad()
    def reset_parameters(self):
        self.centroids.set_(
            self.manifold.random(
                self.centroids.shape, std=self.std / self.centroid_shape[-1] ** 0.5
            )
        )
        self.centroids[0] = 0.0


class Distance2PoincareCentroids2d(Distance2PoincareCentroids):
    def forward(self, input):
        input = input.unsqueeze(-3)
        centroids = self.centroids.permute(1, 0)
        centroids = centroids.view(centroids.shape + (1, 1))
        if self.squared:
            dist = self.manifold.dist2(input, centroids, dim=-4)
        else:
            dist = self.manifold.dist(input, centroids, dim=-4)
        return dist


class WeightedPoincareCentroids(ManifoldModule):
    def __init__(
        self,
        centroid_shape: int,
        num_centroids: int,
        method: str = "einstein",
        *,
        ball,
        learn_origin=True,
        std=1.0
    ):
        super().__init__()

        if not isinstance(num_centroids, int) or num_centroids < 1:
            raise TypeError("num_centroids should be int > 0")
        self.centroid_shape = centroid_shape = geoopt.utils.size2shape(centroid_shape)
        self.std = std
        self.num_centroids = num_centroids
        self.manifold = ball
        self.method = method
        self.centroids = geoopt.ManifoldParameter(
            torch.empty((num_centroids,) + centroid_shape), manifold=ball
        )
        if method == "tangent":
            origin_shape = centroid_shape
        else:
            origin_shape = None
        self.register_origin(
            "origin",
            self.manifold,
            origin_shape=origin_shape,
            allow_none=True,
            parameter=learn_origin,
        )
        self.learn_origin = learn_origin and method == "tangent"
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        self.centroids.set_(
            self.manifold.random(
                self.centroids.shape,
                std=self.std / self.centroid_shape[-1] ** 0.5,
                dtype=self.centroids.dtype,
                device=self.centroids.device,
            )
        )
        self.centroids[0] = 0.0
        if self.origin is not None:
            self.origin.set_(
                self.manifold.origin(
                    self.origin.shape,
                    dtype=self.centroids.dtype,
                    device=self.centroids.device,
                )
            )

    def forward(self, weights):
        return poincare_lincomb(
            self.centroids,
            weights=weights,
            reducedim=-2,
            dim=-1,
            ball=self.manifold,
            keepdim=False,
            method=self.method,
            origin=self.origin,
        )

    def extra_repr(self) -> str:
        return (
            "manifold={self.manifold}, "
            "centroid_shape={centroid_shape}, "
            "num_centroids={num_centroids}, "
            "method={method}, "
            "learn_origin={learn_origin}".format(**self.__dict__, self=self)
        )


class WeightedPoincareCentroids2d(WeightedPoincareCentroids):
    def forward(self, weights):
        if self.origin is not None:
            origin = self.origin.view(*self.origin.shape, 1, 1)
        else:
            origin = None
        return poincare_lincomb(
            self.centroids.view(*self.centroids.shape, 1, 1),
            weights=weights,
            reducedim=-4,
            dim=-3,
            ball=self.manifold,
            keepdim=False,
            method=self.method,
            origin=origin,
        )
