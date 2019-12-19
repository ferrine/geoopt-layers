import torch.nn
import geoopt
from geoopt_layers.base import ManifoldModule
from geoopt_layers.shape import Permuted

__all__ = [
    "TangentLambda",
    "Remap",
    "RemapLambda",
    "Logmap",
    "Expmap",
    "Expmap2d",
    "Logmap2d",
]


class TangentLambda(ManifoldModule):
    """
    Tangent Lambda Layer.

    Applies a custom function in the tangent space around some origin point

    Parameters
    ----------
    fn : callable
        a function to apply in the tangent space
    manifold : geoopt.manifolds.Manifold
        the underlying manifold
    origin : geoopt.ManifoldTensor
        origin point to construct tangent space for the function
    out_origin : Optional[geoopt.ManifoldTensor]
        origin point to use for exponential map
    origin_shape : Tuple[int]|int
        shape of origin point if origin is not provided
    learn_origin : bool
        make origin point trainable? (default True)
    same_origin : bool
        use the same ``out_origin`` as ``origin``
    """

    def __init__(
        self,
        fn,
        manifold: geoopt.manifolds.Manifold,
        origin: geoopt.ManifoldTensor = None,
        out_origin: geoopt.ManifoldTensor = None,
        origin_shape=None,
        learn_origin=True,
        same_origin=True,
    ):
        super().__init__()
        self.manifold = manifold
        self.register_origin("origin", manifold, origin, origin_shape, learn_origin)
        if same_origin and out_origin is None:
            self.register_buffer("out_origin", self.origin)
        else:
            self.register_origin(
                "out_origin", manifold, out_origin, origin_shape, learn_origin
            )
            if self.out_origin.shape != self.origin.shape:
                raise ValueError("Shapes of origins should match")
        self.fn = fn

    def forward(self, input, *other):
        tangent = self.manifold.logmap(self.origin, input)
        out_tangent = self.fn(tangent, *other)
        out_tangent = self.manifold.proju(self.origin, out_tangent)
        if self.out_origin is not self.origin:
            out_tangent = self.manifold.transp(
                self.origin, self.out_origin, out_tangent
            )
        return self.manifold.expmap(self.out_origin, out_tangent)


class RemapLambda(ManifoldModule):
    """
    Remap Tangent Lambda Layer.

    Perform nonlinear transformation in tangent space of one manifold and then map the result into the new
    manifold. New manifold may me exactly the same. In this case some restrictions
    are applied and parallel transport performed.

    Parameters
    ----------
    fn : callable
        function to apply in tangent space
    source_manifold : geoopt.manifolds.Manifold
        input manifold
    target_manifold : Optional[geoopt.manifolds.Manifold]
        output manifold
    source_origin : Optional[geoopt.ManifoldTensor]
        origin point to construct tangent space
    target_origin : Optional[geoopt.ManifoldTensor]
        origin point to perform final exponential map
    source_origin_shape : Tuple[int]|int
        shape of source origin point if tensor is not provided
    target_origin_shape : Tuple[int]|int
        shape of target origin point if tensor is not provided
    learn_origin : bool
        make origin point trainable? (default True)
    """

    def __init__(
        self,
        fn,
        source_manifold: geoopt.manifolds.Manifold,
        target_manifold: geoopt.manifolds.Manifold = None,
        source_origin: geoopt.ManifoldTensor = None,
        target_origin: geoopt.ManifoldTensor = None,
        source_origin_shape=None,
        target_origin_shape=None,
        learn_origin=True,
    ):
        super().__init__()
        if target_manifold is None:
            target_manifold = source_manifold
            target_origin_shape = (
                target_origin_shape or source_origin_shape or source_origin.shape
            )
        self.source_manifold = source_manifold
        self.target_manifold = target_manifold

        self.register_origin(
            "source_origin",
            source_manifold,
            source_origin,
            source_origin_shape,
            learn_origin,
        )
        self.register_origin(
            "target_origin",
            target_manifold,
            target_origin,
            target_origin_shape,
            learn_origin,
        )
        if self.source_manifold is self.target_manifold:
            if not self.source_origin.shape == self.target_origin.shape:
                raise ValueError(
                    "When remapping on the same manifold can't have different origin shapes"
                )
        self.fn = fn

    def forward(self, input, *other):
        tangent = self.source_manifold.logmap(self.source_origin, input)
        out_tangent = self.fn(tangent, *other)
        if self.source_manifold is self.target_manifold:
            out_tangent = self.source_manifold.transp(
                self.source_origin, self.target_origin, out_tangent
            )
        return self.target_manifold.expmap(self.target_origin, out_tangent)


class Remap(RemapLambda):
    """
    Remap Layer.

    Remap all points from one origin/manifold to another.

    Parameters
    ----------
    source_manifold : geoopt.manifolds.Manifold
        input manifold
    target_manifold : Optional[geoopt.manifolds.Manifold]
        output manifold
    source_origin : Optional[geoopt.ManifoldTensor]
        origin point to construct tangent space
    target_origin : Optional[geoopt.ManifoldTensor]
        origin point to perform final exponential map
    source_origin_shape : Tuple[int]|int
        shape of source origin point if tensor is not provided
    target_origin_shape : Tuple[int]|int
        shape of target origin point if tensor is not provided
    learn_origin: bool
        make origin point trainable? (default True)
    """

    def __init__(
        self,
        source_manifold: geoopt.manifolds.Manifold,
        target_manifold: geoopt.manifolds.Manifold = None,
        source_origin: geoopt.ManifoldTensor = None,
        target_origin: geoopt.ManifoldTensor = None,
        source_origin_shape=None,
        target_origin_shape=None,
        learn_origin=True,
    ):
        super().__init__(
            fn=torch.nn.Identity(),
            source_manifold=source_manifold,
            target_manifold=target_manifold,
            source_origin=source_origin,
            target_origin=target_origin,
            source_origin_shape=source_origin_shape,
            target_origin_shape=target_origin_shape,
            learn_origin=learn_origin,
        )


class Expmap(ManifoldModule):
    def __init__(
        self,
        manifold: geoopt.manifolds.Manifold,
        origin=None,
        origin_shape=None,
        learn_origin=True,
    ):
        super().__init__()
        self.manifold = manifold
        self.register_origin("origin", manifold, origin, origin_shape, learn_origin)

    def forward(self, input):
        input = self.manifold.proju(self.origin, input)
        return self.manifold.expmap(self.origin, input)


class Logmap(ManifoldModule):
    def __init__(
        self,
        manifold: geoopt.manifolds.Manifold,
        origin=None,
        origin_shape=None,
        learn_origin=True,
    ):
        super().__init__()
        self.manifold = manifold
        self.register_origin("origin", manifold, origin, origin_shape, learn_origin)

    def forward(self, input):
        return self.manifold.logmap(self.origin, input)


class Expmap2d(Permuted, ManifoldModule):
    def __init__(
        self,
        manifold: geoopt.manifolds.Manifold,
        origin=None,
        origin_shape=None,
        learn_origin=True,
        contiguous=True,
    ):
        if not manifold.ndim == 1:
            raise ValueError("manifold.ndim is required to be 1")
        expmap = Expmap(
            manifold,
            origin=origin,
            origin_shape=origin_shape,
            learn_origin=learn_origin,
        )
        super().__init__(expmap, (0, 2, 3, 1), contiguous=contiguous)


class Logmap2d(Permuted, ManifoldModule):
    def __init__(
        self,
        manifold: geoopt.manifolds.Manifold,
        origin=None,
        origin_shape=None,
        learn_origin=True,
        contiguous=True,
    ):
        if not manifold.ndim == 1:
            raise ValueError("manifold.ndim is required to be 1")
        logmap = Logmap(
            manifold,
            origin=origin,
            origin_shape=origin_shape,
            learn_origin=learn_origin,
        )
        super().__init__(logmap, (0, 2, 3, 1), contiguous=contiguous)
