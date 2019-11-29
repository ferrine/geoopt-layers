import geoopt.utils
import functools
import operator
import torch

__all__ = ["ManifoldModule", "prod", "idx2sign", "reshape_shape"]


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


def prod(items):
    return functools.reduce(operator.mul, items, 1)


def reshape_shape(shape, pattern):
    out_shape = []
    i = 0
    for s in pattern:
        if isinstance(s, int):
            break
        out_shape.append(shape[i])
        i += 1
    for s in pattern[i:]:
        if s == "*":
            break
        out_shape.append(pattern[i])
        i += 1

    for j, s in enumerate(reversed(pattern[i:])):
        out_shape.insert(i, shape[-j - 1])

    return out_shape


def idx2sign(idx, dim, neg=True):
    """
    Unify idx to be negative or positive, that helps in cases of broadcasting.

    Parameters
    ----------
    idx : int
        current index
    dim : int
        maximum dimension
    neg : bool
        indicate we need negative index

    Returns
    -------
    int
    """
    if neg:
        if idx < 0:
            return idx
        else:
            return (idx + 1) % -(dim + 1)
    else:
        return idx % dim
