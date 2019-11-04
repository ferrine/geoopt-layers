import geoopt


def create_origin(
    manifold: geoopt.manifolds.Manifold,
    origin: geoopt.ManifoldTensor = None,
    origin_shape=None,
    parameter=False,
):
    if origin is not None:
        manifold.assert_attached(origin)
    elif origin_shape is None:
        raise ValueError(
            "`origin_shape` is the required parameter if origin is not provided"
        )
    else:
        origin = manifold.origin(origin_shape)
    if parameter:
        origin = geoopt.ManifoldParameter(origin)
    return origin
