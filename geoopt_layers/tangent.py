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
            origin = geoopt.ManifoldParameter(origin)
        if out_origin is None:
            out_origin = origin
        elif learn_origin:
            out_origin = geoopt.ManifoldParameter(out_origin)
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
    def __init__(
        self,
        fn,
        source_manifold: geoopt.manifolds.Manifold,
        target_manifold: geoopt.manifolds.Manifold,
        source_origin: geoopt.ManifoldTensor = None,
        target_origin: geoopt.ManifoldTensor = None,
        source_origin_shape=None,
        target_origin_shape=None,
        learn_origin=True,
    ):
        super().__init__()

        # source manifold
        self.source_manifold = source_manifold
        if source_origin is not None:
            self.source_manifold.assert_attached(source_origin)
        elif source_origin_shape is None:
            raise ValueError(
                "source origin shape is the required parameter if origin is not provided"
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
                "target origin shape is the required parameter if origin is not provided"
            )
        else:
            target_origin = self.target_manifold.origin(target_origin_shape)
        if learn_origin:
            target_origin = geoopt.ManifoldParameter(
                target_origin, manifold=target_manifold
            )
        self.target_origin = target_origin

        self.fn = fn

    def forward(self, input):
        self.source_manifold.assert_attached(input)
        tangent = self.source_manifold.logmap(self.source_origin, input)
        out_tangent = self.fn(tangent)
        return self.target_manifold.expmap(self.target_origin, out_tangent)
