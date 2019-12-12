from typing import Tuple, Union
import geoopt.utils
from .utils import idx2sign, prod
from geoopt_layers.base import ManifoldModule
from .functional import pairwise_distances
import torch

__all__ = ["Distance2Centroids", "PairwiseDistances", "KNN", "KNNIndex"]


class Distance2Centroids(ManifoldModule):
    """
    Distance Layer.

    Computes distances to centroids for input. Centroids should be atomic points on the manifold.
    In case you desire product space for centroids, explicitly provide ProductManifold for centroids.

    Parameters
    ----------
    manifold : geoopt.manifolds.Manifold
        manifold to operate on
    centroid_shape : Union[int, Tuple[int]]
        shape of a single centroid
    num_centroids : int
        number of centroids
    squared : bool
        compute squared distance? (default: True)
    """

    def __init__(
        self,
        manifold: geoopt.manifolds.Manifold,
        centroid_shape: Union[int, Tuple[int]],
        num_centroids: int,
        squared=False,
    ):
        super().__init__()

        if not isinstance(num_centroids, int) or num_centroids < 1:
            raise TypeError("num_centroids should be int > 0")
        self.centroid_shape = centroid_shape = geoopt.utils.size2shape(centroid_shape)
        if len(centroid_shape) != manifold.ndim:
            raise ValueError(
                "shape of centroid should be of minimum atomic size, {} for {}".format(
                    manifold.ndim, manifold
                )
            )
        self.num_centroids = num_centroids
        self.manifold = manifold
        self.centroids = geoopt.ManifoldParameter(
            torch.empty((num_centroids,) + centroid_shape), manifold=manifold
        )
        self.squared = squared
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        self.centroids.set_(
            self.manifold.random(
                self.centroids.shape,
                dtype=self.centroids.dtype,
                device=self.centroids.device,
            )
        )

    def forward(self, input):

        input = input.unsqueeze(-self.manifold.ndim - 1)
        if self.squared:
            output = self.manifold.dist2(self.centroids, input)
        else:
            output = self.manifold.dist(self.centroids, input)
        return output

    def extra_repr(self) -> str:
        return (
            "manifold={self.manifold}, "
            "centroid_shape={centroid_shape}, "
            "num_centroids={num_centroids}, "
            "squared={squared}".format(**self.__dict__, self=self)
        )


class PairwiseDistances(ManifoldModule):
    """
    Pairwise Distance Layer.

    Compute pairwise distances between points.

    Parameters
    ----------
    manifold : Optional[geoopt.manifold.Manifold]
    dim : int
        compute pairwise distance for this shape
    squared : True
        compute squared distance? (default: True)
    """

    def __init__(self, manifold, dim: int, squared=False):
        super().__init__()
        self.squared = squared
        if dim == -1:
            raise ValueError("dim should be not the last one")
        self.dim = dim
        self.manifold = manifold

    def forward(self, x, y=None):
        return pairwise_distances(
            x=x, y=y, dim=self.dim, squared=self.squared, manifold=self.manifold
        )

    def extra_repr(self) -> str:
        return "manifold={self.manifold}, squared={squared}".format(
            **self.__dict__, self=self
        )


class KNNIndex(PairwiseDistances):
    """
    K Nearest Neighbors on the manifold.


    """

    def __init__(self, manifold, k: int, dim: int, squared: bool = False, unroll=None):
        super().__init__(manifold=manifold, dim=dim, squared=squared)
        self.k = k
        self.unroll = unroll

    @torch.no_grad()
    def forward(self, x, y=None):
        if self.dim < 0:
            dim = self.dim + self.manifold.ndim
        else:
            dim = self.dim + 1
        if self.unroll is None:
            distances = super().forward(x, y)
            _, idx = distances.topk(k=self.k, dim=dim, largest=False)
            return idx
        else:
            shape_x = list(x.shape)
            shape_y = list(y.shape)
            shape_y[self.dim] = self.k
            shape_y.insert(self.dim, 1)
            shape_x.insert(self.dim + 1, 1)
            result_shape = geoopt.utils.broadcast_shapes(shape_x, shape_y)
            result_shape = result_shape[: -self.manifold.ndim]
            idx_out = torch.empty(result_shape, dtype=torch.int64, device=x.device)
            for i, (mini_x, mini_y) in enumerate(
                zip(x.unbind(self.unroll), y.unbind(self.unroll))
            ):
                slc = tuple(
                    slice(None) if d != self.unroll else i for d in range(idx_out.dim())
                )
                idx_out[slc] = (
                    super()
                    .forward(
                        mini_x.unsqueeze(self.unroll), mini_y.unsqueeze(self.unroll)
                    )
                    .topk(k=self.k, dim=dim, largest=False)[1]
                )
            return idx_out


class KNN(KNNIndex):
    def __init__(self, manifold, k: int, dim: int, squared: bool = False, unroll=None):
        super().__init__(
            manifold=manifold, k=k, dim=dim, squared=squared, unroll=unroll
        )

    def forward(self, x, y=None):
        if y is None:
            y = x
        idx = super().forward(x, y)
        dim = idx2sign(self.dim, idx.dim())
        y = y.unsqueeze(dim)
        idx = idx.view(idx.shape + (1,) * self.manifold.ndim)
        y, idx = torch.broadcast_tensors(y, idx)
        top_k_out = torch.gather(y, dim - 1, idx)
        return top_k_out
