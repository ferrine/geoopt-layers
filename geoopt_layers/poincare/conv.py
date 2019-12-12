import torch.nn
from .functional import mobius_conv2d
from torch.nn.modules.utils import _pair
from ..base import ManifoldModule
from ..utils import prod


__all__ = ["MobiusConv2d"]


class MobiusConv2d(ManifoldModule):
    """
    Hyperbolic convolution.

    Notes
    -----
    Shapes: Bx(D*P)xHxW, originally reshaped from BxDxPxxHxW

    Parameters
    ----------
    dim : int
        dimension input of Poincare Ball
    dim_out : int
        dimension output of Poincare Ball
    kernel_size : int|tuple
        Convolutions kernel size
    stride : int|tuple
        Convolution stride
    padding : int|tuple
        Convolution padding (padded with zeros)
    dilation : int|tuple
        Convolution dilation
    points_in : int
        Number of points in input tensor (default 1)
    points_out : int
        Number of points in output tensor (default 1)
    ball : geoopt.PoincareBall
        input Poincare Ball manifold
    ball_out : geoopt.PoincareBall
        output Poincare Ball manifold (defaults to input Ball)
    """

    def __init__(
        self,
        dim,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        *,
        dim_out=None,
        points_in=1,
        points_out=1,
        ball,
        ball_out=None,
        matmul=True
    ):
        super().__init__()
        self.ball = ball
        if ball_out is None:
            ball_out = ball
        if dim_out is None:
            dim_out = dim
        self.ball_out = ball_out
        self.dim = dim
        self.dim_out = dim_out
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.points_in = points_in
        self.points_out = points_out
        self.matmul = matmul
        if self.matmul:
            self.weight_mm = torch.nn.Parameter(
                torch.empty(dim_out * points_in, dim * points_in), requires_grad=True
            )
        else:
            self.register_parameter("weight_mm", None)
            if ball is not ball_out:
                raise ValueError(
                    "If not performing matmul, output_ball should be same as input ball"
                )
            if dim_out != dim:
                raise ValueError(
                    "If not performing matmul, dim_out ({}) should be same as dim {}".format(
                        dim_out, dim
                    )
                )
        self.weight_avg = torch.nn.Parameter(
            torch.empty(points_out, points_in, *self.kernel_size), requires_grad=True
        )
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        if self.weight_mm is not None:
            torch.nn.init.eye_(self.weight_mm)
            self.weight_mm.add_(torch.empty_like(self.weight_mm).normal_(0, 1e-3))
        self.weight_avg.fill_(1 / prod(self.kernel_size))

    def forward(self, input):
        return mobius_conv2d(
            input,
            weight_mm=self.weight_mm,
            weight_avg=self.weight_avg,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            points_in=self.points_in,
            points_out=self.points_out,
            ball=self.ball,
            ball_out=self.ball_out,
        )

    def extra_repr(self) -> str:
        return (
            "dim_in={dim}, "
            "dim_out={dim_out}, "
            "kernel_size={kernel_size}, "
            "stride={stride}, "
            "padding={padding}, "
            "dilation={dilation}, "
            "points_in={points_in}, "
            "points_out={points_out}, "
            "matmul={matmul}"
        ).format(**self.__dict__)
