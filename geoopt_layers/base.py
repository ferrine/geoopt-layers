import geoopt.manifolds
import torch


def create_origin(
    manifold: geoopt.manifolds.Manifold,
    origin: geoopt.ManifoldTensor = None,
    origin_shape=None,
    parameter=False,
    allow_none=False,
):
    if origin is not None:
        pass
    elif origin_shape is None and not allow_none:
        raise ValueError(
            "`origin_shape` is the required parameter if origin is not provided"
        )
    elif origin_shape is None and allow_none:
        return None
    else:
        origin = manifold.origin(origin_shape)
    if parameter:
        origin = geoopt.ManifoldParameter(origin)
    return origin


class ManifoldModule(torch.nn.Module):
    def register_origin(
        self,
        name: str,
        manifold: geoopt.manifolds.Manifold,
        origin: geoopt.ManifoldTensor = None,
        origin_shape=None,
        parameter=False,
        allow_none=False,
    ):
        if manifold not in set(self.children()):
            raise ValueError(
                "Manifold should be a child of module to register an origin"
            )
        origin = create_origin(
            manifold=manifold,
            origin=origin,
            origin_shape=origin_shape,
            parameter=parameter,
            allow_none=allow_none,
        )
        if isinstance(origin, torch.nn.Parameter):
            self.register_parameter(name, origin)
        else:
            self.register_buffer(name, origin)
