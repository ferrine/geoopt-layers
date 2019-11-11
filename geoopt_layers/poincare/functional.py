import geoopt
import torch.nn.functional

from geoopt_layers.poincare import math


def mobius_adaptive_max_pool2d(input, output_size, return_indices=False):
    norms = input.norm(dim=1, keepdim=True)
    _, idx = torch.nn.functional.adaptive_max_pool2d(
        norms, output_size, return_indices=True
    )
    out = input.view(input.shape[0], input.shape[1], -1)
    out = out[
        torch.arange(input.shape[0], device=input.device).view((-1, 1, 1, 1)),
        torch.arange(input.shape[1], device=input.device).view((1, -1, 1, 1)),
        idx,
    ]
    out = out.permute(0, 2, 3, 1)
    if return_indices:
        idx = idx.permute(0, 2, 3, 1)
        return out, idx
    else:
        return out


def mobius_max_pool2d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
    return_indices=False,
):
    norms = input.norm(dim=1, keepdim=True)
    _, idx = torch.nn.functional.max_pool2d(
        norms, kernel_size, stride, padding, dilation, ceil_mode, return_indices=True
    )
    out = input.view(input.shape[0], input.shape[1], -1)
    out = out[
        torch.arange(input.shape[0], device=input.device).view((-1, 1, 1, 1)),
        torch.arange(input.shape[1], device=input.device).view((1, -1, 1, 1)),
        idx,
    ]
    if return_indices:
        return out, idx
    else:
        return out


def mobius_avg_pool2d(
    input, kernel_size, stride=None, padding=0, ceil_mode=False, *, ball
):
    input = math.poincare2klein(input, dim=1, c=ball.c)
    lorentz = math.lorentz_factor(input, dim=1, keepdim=True, c=ball.c)
    input_avg = torch.nn.functional.avg_pool2d(
        input * lorentz,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=False,
    )
    lorentz_avg = torch.nn.functional.avg_pool2d(
        lorentz,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=False,
    )
    output = input_avg / lorentz_avg
    output = math.klein2poincare(output, c=ball.c, dim=1)
    output = ball.projx(output, dim=1)
    return output


def mobius_adaptive_avg_pool2d(input, output_size, *, ball: geoopt.PoincareBall):
    input = math.poincare2klein(input, c=ball.c, dim=1)
    lorentz = math.lorentz_factor(input, dim=1, keepdim=True, c=ball.c)
    input_avg = torch.nn.functional.adaptive_avg_pool2d(
        input * lorentz, output_size=output_size
    )
    lorentz_avg = torch.nn.functional.adaptive_avg_pool2d(
        lorentz, output_size=output_size
    )
    output = input_avg / lorentz_avg
    output = math.klein2poincare(output, c=ball.c, dim=1)
    output = ball.projx(output, dim=1)
    return output
