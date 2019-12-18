import geoopt
from geoopt_layers.base import ManifoldModule
from .. import distance
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


@torch.no_grad()
def keep_zero(mod, input):
    mod.centroids[0] = 0.0


class Distance2PoincareCentroids(distance.Distance2Centroids):
    n = 0

    def __init__(
        self,
        centroid_shape: int,
        num_centroids: int,
        squared=False,
        *,
        ball,
        std=1.0,
        zero=False,
    ):
        self.std = std
        super().__init__(
            ball,
            centroid_shape=centroid_shape,
            num_centroids=num_centroids,
            squared=squared,
        )
        self.zero = zero
        if zero:
            self.register_forward_pre_hook(keep_zero)

    @torch.no_grad()
    def reset_parameters(self):
        self.centroids.set_(self.manifold.random(self.centroids.shape, std=self.std))
        self.centroids[0] = 0.0

    def extra_repr(self) -> str:
        text = super().extra_repr()
        text += ", zero={}".format(self.zero)
        return text

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
        zero=False,
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
        self.zero = zero
        if zero:
            self.register_forward_pre_hook(keep_zero)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        self.centroids.set_(
            self.manifold.random(
                self.centroids.shape,
                std=self.std,
                dtype=self.centroids.dtype,
                device=self.centroids.device,
            )
        )
        self.centroids[0] = 0.0
        if self.origin is not None:
            self.origin.zero_()

    def forward(self, weights):
        if self.origin is not None:
            origin = self.origin.view(self.origin.shape + (1,) * self.n)
        else:
            origin = None
        return poincare_mean(
            self.centroids.view(self.centroids.shape + (1,) * self.n),
            weights=weights,
            reducedim=-self.n - 2,
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
            "lincomb={lincomb}, "
            "zero={zero}".format(**self.__dict__, self=self)
        )


class WeightedPoincareCentroids1d(WeightedPoincareCentroids):
    n = 1


class WeightedPoincareCentroids2d(WeightedPoincareCentroids):
    n = 2


class WeightedPoincareCentroids3d(WeightedPoincareCentroids):
    n = 3
