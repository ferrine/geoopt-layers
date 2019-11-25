import geoopt.utils
import functools
import operator
import torch

__all__ = ["Permute", "Permuted", "ManifoldModule", "prod", "Reshape"]


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


class Permute(torch.nn.Module):
    """
    Permute Layer.

    All manifold aware layers assume manifold dimension is the last dimension.
    Therefore, if you want to use regular pytorch layers within geoopt-layers,
    you may need (like in case 2d convolutions) permute dimensions

    Parameters
    ----------
    permutation : int|tuple
        parameters for pytorch permute
    contiguous : bool
        make result contiguous if True (default False)
    """

    def __init__(self, *permutation, contiguous=False):
        super().__init__()
        self.permutation = geoopt.utils.size2shape(*permutation)
        self.contiguous = contiguous

    def forward(self, input):
        out = input.permute(*self.permutation)
        if self.contiguous:
            out = out.contiguous()
        return out

    def inverse(self, contiguous=None):
        """
        Invert the permutation.

        Parameters
        ----------
        contiguous : Optional[bool]
            make result contiguous if True, defaults to instance contiguous parameter

        Returns
        -------
        Permute
        """
        reverse_permute = [
            self.permutation.index(l) for l in range(len(self.permutation))
        ]
        if contiguous is None:
            contiguous = self.contiguous
        return self.__class__(*reverse_permute, contiguous=contiguous)

    def extra_repr(self) -> str:
        return " ".join(map(str, self.permutation))


class Permuted(torch.nn.Module):
    """
    Permuted function.

    Applies a function with input permutation, permutes the result back after.
    """

    def __init__(self, function, permutation, contiguous=False):
        super().__init__()
        self.forward_permutation = Permute(*permutation, contiguous=contiguous)
        self.reverse_permutation = self.forward_permutation.inverse()
        self.function = function

    def forward(self, input):
        input = self.forward_permutation(input)
        output = self.function(input)
        output = self.reverse_permutation(output)
        return output

    def extra_repr(self) -> str:
        return " ".join(map(str, self.forward_permutation.permutation))


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


class Reshape(torch.nn.Module):
    def __init__(self, *pattern):
        super().__init__()
        self.pattern = list(pattern)

    def forward(self, input: torch.Tensor):
        out_shape = reshape_shape(input.shape, self.pattern)
        return input.reshape(out_shape)

    def extra_repr(self) -> str:
        return " ".join(map(str, self.pattern))
