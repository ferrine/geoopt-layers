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
    if return_indices:
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


def mobius_adaptive_avg_pool2d(input, output_size, *, ball):
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


def mobius_batch_norm2d(
    input,
    running_midpoint,
    running_variance,
    beta1,
    beta2,
    alpha=1.0,
    epsilon=1e-4,
    training=True,
    *,
    ball,
):
    if training:
        midpoint = math.poincare_mean(input, dim=1, keepdim=True, c=ball.c)
        midpoint = ball.projx(midpoint, dim=1)
        variance = ball.dist(midpoint, input, dim=1).pow(2).mean()

        input = ball.mobius_add(-midpoint, input, dim=1)
        input = ball.mobius_scalar_mul(
            alpha / (variance + epsilon) ** 0.5, input, dim=1
        )
        with torch.no_grad():
            running_variance.mul_(beta1).add_(1 - beta1, variance)
            running_midpoint.set_(
                ball.geodesic(beta2, midpoint, running_midpoint, dim=1).data
            )
    else:
        input = ball.mobius_add(-running_midpoint, input, dim=1)
        input = ball.mobius_scalar_mul(
            alpha / (running_variance + epsilon) ** 0.5, input, dim=1
        )
    return ball.attach(input)


def mobius_linear(
    input,
    weight,
    bias=None,
    *,
    ball: geoopt.PoincareBall,
    ball_out: geoopt.PoincareBall,
    source_origin=None,
    target_origin=None,
):
    if source_origin is not None and target_origin is not None:
        # We need to take care of origins
        tangent = ball.logmap(source_origin, input)
        new_tangent = tangent @ weight
        if ball is ball_out:
            # In case same manifolds are used, we need to perform parallel transport
            new_tangent = ball.transp(source_origin, target_origin, new_tangent)
        output = ball_out.expmap(target_origin, new_tangent)
        if bias is not None:
            output = ball_out.mobius_add(output, bias)
    else:
        if ball is ball_out:
            output = ball.mobius_matvec(weight, input)
            if bias is not None:
                output = ball.mobius_add(output, bias)
        else:
            tangent = ball.logmap0(input)
            new_tangent = tangent @ weight
            output = ball_out.expmap0(new_tangent)
            if bias is not None:
                output = ball_out.mobius_add(output, bias)
    return output
