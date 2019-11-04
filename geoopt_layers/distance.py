from typing import Tuple, Union
import geoopt.utils
import torch


class Distance2Centroids(torch.nn.Module):
    """
    Distance Layer.

    Computes distances to centroids for input. Centroids should be atomic points on the manifold.
    In case you desire product space for centroids, explicitly provide ProductManifold for centroids.

    Parameters
    ----------
    manifold : geoopt.manifolds.Manifold
    centroid_shape : Union[int, Tuple[int]]
    num_centroids : int
    squared : bool
    """

    def __init__(
        self,
        manifold: geoopt.manifolds.Manifold,
        centroid_shape: Union[int, Tuple[int]],
        num_centroids: int,
        squared=True,
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

    @torch.no_grad()
    def reset_parameters(self):
        self.centroids.set_(self.manifold.random(self.centroids.shape))

    def forward(self, input):
        self.manifold.assert_attached(input)
        if self.manifold.ndim > 0:
            input = input.unsqueeze(-self.manifold.ndim - 1)
        if self.squared:
            output = self.manifold.dist2(self.centroids, input)
        else:
            output = self.manifold.dist(self.centroids, input)
        return output
