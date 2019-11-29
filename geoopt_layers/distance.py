from typing import Tuple, Union
import geoopt.utils
from .utils import ManifoldModule, idx2sign, prod
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

    def __init__(
        self, manifold, k: int, dim: int, squared: bool = False, return_distances=False
    ):
        super().__init__(manifold=manifold, dim=dim, squared=squared)
        self.k = k
        self.return_distances = return_distances

    @torch.no_grad()
    def forward(self, x, y=None):
        distances = super().forward(x, y)
        dim = idx2sign(self.dim, distances.dim()) + 1
        distances, idx = distances.topk(k=self.k, dim=dim, largest=False)
        if self.return_distances:
            return distances, idx
        else:
            return idx


def swap_end_dims(x, *dims):
    for i, dim in zip(range(-len(dims), 0), dims):
        j = idx2sign(dim, x.dim())
        if j == i:
            continue
        else:
            x = x.transpose(i, j)
    return x


class KNN(KNNIndex):
    def __init__(self, manifold, k: int, dim: int, squared: bool = False):
        super().__init__(manifold=manifold, k=k, dim=dim, squared=squared)

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
