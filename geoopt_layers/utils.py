import geoopt.utils
import torch

__all__ = ["Permute", "Permuted", "ManifoldModule"]


def create_origin(
    manifold: geoopt.manifolds.Manifold,
    origin: geoopt.ManifoldTensor = None,
    origin_shape=None,
    parameter=False,
    allow_none=False,
):
    if origin is not None:
        manifold.assert_attached(origin)
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
    manifold : Optional[geoopt.manifolds.Manifold]
        Optionally attach the manifold to the output
    """

    def __init__(self, *permutation, manifold=None, contiguous=True):
        super().__init__()
        self.permutation = geoopt.utils.size2shape(*permutation)
        self.manifold = manifold
        self.contiguous = contiguous

    def forward(self, input):
        out = input.permute(*self.permutation)
        if self.contiguous:
            out = out.contiguous()
        if self.manifold is not None:
            out = self.manifold.attach(out)
        return out

    def inverse(self, manifold=None):
        """
        Invert the permutation

        Parameters
        ----------
        manifold : Optional[geoopt.manifolds.Manifold]
            Optionally attach the manifold to the output

        Returns
        -------
        Permute
        """
        reverse_permute = [
            self.permutation.index(l) for l in range(len(self.permutation))
        ]
        return self.__class__(
            *reverse_permute, manifold=manifold, contiguous=self.contiguous
        )


class Permuted(torch.nn.Module):
    """
    Permuted function.

    Applies a function with input permutation, permutes the result back after.
    """

    def __init__(
        self,
        function,
        permutation,
        input_manifold=None,
        output_manifold=None,
        contiguous=True,
    ):
        super().__init__()
        self.forward_permutation = Permute(
            *permutation, contiguous=contiguous, manifold=input_manifold
        )
        self.reverse_permutation = self.forward_permutation.inverse(
            manifold=output_manifold
        )
        self.function = function

    def forward(self, input):
        input = self.forward_permutation(input)
        output = self.function(input)
        output = self.reverse_permutation(output)
        return output
