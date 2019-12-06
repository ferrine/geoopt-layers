import torch.nn.functional
from geoopt_layers.base import ManifoldModule
from geoopt_layers.poincare.functional import (
    mobius_adaptive_max_pool2d,
    mobius_max_pool2d,
    mobius_avg_pool2d,
    mobius_adaptive_avg_pool2d,
)

__all__ = [
    "MobiusAvgPool2d",
    "MobiusAdaptiveAvgPool2d",
    "MobiusMaxPool2d",
    "MobiusAdaptiveMaxPool2d",
]


class MobiusAvgPool2d(torch.nn.AvgPool2d, ManifoldModule):
    def __init__(self, kernel_size, stride=1, padding=0, ceil_mode=False, *, ball):
        super().__init__(
            kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode
        )
        self.ball = ball

    def forward(self, input):

        return mobius_avg_pool2d(
            input,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
            ball=self.ball,
        )


class MobiusAdaptiveAvgPool2d(torch.nn.AdaptiveAvgPool2d, ManifoldModule):
    def __init__(self, output_size, *, ball):
        super().__init__(output_size)
        self.ball = ball

    def forward(self, input):
        return mobius_adaptive_avg_pool2d(
            input, output_size=self.output_size, ball=self.ball
        )


class MobiusMaxPool2d(torch.nn.MaxPool2d, ManifoldModule):
    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        dilation=1,
        return_indices=False,
        ceil_mode=False,
        *,
        ball
    ):
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode,
        )
        self.ball = ball

    def forward(self, input: torch.Tensor):
        return mobius_max_pool2d(
            input,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=self.return_indices,
        )


class MobiusAdaptiveMaxPool2d(torch.nn.AdaptiveMaxPool2d, ManifoldModule):
    def __init__(self, output_size, return_indices=False, *, ball):
        super().__init__(output_size=output_size, return_indices=return_indices)
        self.ball = ball

    def forward(self, input: torch.Tensor):
        return mobius_adaptive_max_pool2d(input, self.output_size, self.return_indices)
