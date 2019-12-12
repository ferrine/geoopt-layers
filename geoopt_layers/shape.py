import geoopt.utils
import torch

from geoopt_layers.utils import reshape_shape


__all__ = ["Reshape", "Permute", "Permuted"]


class Reshape(torch.nn.Module):
    def __init__(self, *pattern):
        super().__init__()
        self.pattern = list(pattern)

    def forward(self, input: torch.Tensor):
        out_shape = reshape_shape(input.shape, self.pattern)
        return input.reshape(out_shape)

    def extra_repr(self) -> str:
        return " ".join(map(str, self.pattern))


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
