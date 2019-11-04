import torch.nn
import geoopt


class TangentLambda(torch.nn.Module):
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
        if origin is not None:
            self.manifold.assert_attached(origin)
        elif origin_shape is None:
            raise ValueError(
                "origin shape is the required parameter id origin is not provided"
            )
        else:
            origin = self.manifold.origin(origin_shape)
        if out_origin is not None:
            self.manifold.assert_attached(out_origin)
            if not out_origin.shape == origin.shape:
                raise ValueError("TangentLambda can't change shape")
        elif not same_origin:
            out_origin = self.manifold.origin(origin_shape)
        if learn_origin:
            origin = geoopt.ManifoldParameter(origin, manifold=manifold)
        if out_origin is None:
            out_origin = origin
        elif learn_origin:
            out_origin = geoopt.ManifoldParameter(out_origin, manifold=manifold)
        self.origin = origin
        self.out_origin = out_origin
        self.fn = fn

    def forward(self, input):
        self.manifold.assert_attached(input)
        tangent = self.manifold.logmap(self.origin, input)
        out_tangent = self.fn(tangent)
        out_tangent = self.manifold.proju(self.origin, out_tangent)
        if self.out_origin is not self.origin:
            out_tangent = self.manifold.transp(
                self.origin, self.out_origin, out_tangent
            )
        return self.manifold.expmap(self.out_origin, out_tangent)


class RemapLambda(torch.nn.Module):
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
        # source manifold
        self.source_manifold = source_manifold
        if source_origin is not None:
            self.source_manifold.assert_attached(source_origin)
        elif source_origin_shape is None:
            raise ValueError(
                "`source_origin_shape` is the required parameter if origin is not provided"
            )
        else:
            source_origin = self.source_manifold.origin(source_origin_shape)
        if learn_origin:
            source_origin = geoopt.ManifoldParameter(
                source_origin, manifold=source_manifold
            )
        self.source_origin = source_origin

        # target manifold
        self.target_manifold = target_manifold
        if target_origin is not None:
            self.target_manifold.assert_attached(target_origin)
        elif target_origin_shape is None:
            raise ValueError(
                "`target_origin_shape` is the required parameter if origin is not provided and manifolds are different"
            )
        else:
            target_origin = self.target_manifold.origin(target_origin_shape)
        if learn_origin:
            target_origin = geoopt.ManifoldParameter(
                target_origin, manifold=target_manifold
            )
        self.target_origin = target_origin
        if source_manifold is target_manifold:
            if not self.source_origin.shape == self.target_origin.shape:
                raise ValueError(
                    "When remapping on the same manifold can't have different origin shapes"
                )
        self.fn = fn

    def forward(self, input):
        self.source_manifold.assert_attached(input)
        tangent = self.source_manifold.logmap(self.source_origin, input)
        out_tangent = self.fn(tangent)
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
