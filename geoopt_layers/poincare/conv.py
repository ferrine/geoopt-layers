import torch.nn
from .functional import mobius_conv2d
from torch.nn.modules.utils import _pair
from ..base import ManifoldModule


__all__ = ["MobiusConv2d"]


class MobiusConv2d(ManifoldModule):
    def __init__(
        self,
        dim_in,
        dim_out,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        *,
        points_in=1,
        points_out=1,
        ball,
        ball_out=None
    ):
        super().__init__()
        self.ball = ball
        if ball_out is None:
            ball_out = ball
        self.out_ball = ball_out
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.points_in = points_in
        self.points_out = points_out
        self.weight_mm = torch.nn.Parameter(
            torch.empty(dim_out * points_in, dim_in * points_in), requires_grad=True
        )
        self.weight_avg = torch.nn.Parameter(
            torch.empty(points_out, points_in, *self.kernel_size), requires_grad=True
        )

    def forward(self, input):
        return mobius_conv2d(
            input,
            weight_mm=self.weight_mm,
            weight_avg=self.weight_avg,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            points_in=self.points,
            points_out=self.points_out,
            ball=self.ball,
            ball_out=self.ball_out,
        )

    def extra_repr(self) -> str:
        return (
            "dim_in={dim_in}, "
            "dim_out={dim_out}, "
            "kernel_size={kernel_size}, "
            "stride={stride}, "
            "padding={padding}, "
            "dilation={dilation}, "
            "points_in={points_in}, "
            "points_out={points_out}"
        ).format(**self.__dict__)

    @torch.no_grad()
    def init_parameters(self):
        torch.nn.init.eye_(self.weight_mm)
        self.weight_avg.fill_(1.)
