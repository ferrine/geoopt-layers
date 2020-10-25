import torch.nn.functional
import functools
import geoopt.utils


def mobius_adaptive_max_pool2d(input, output_size, return_indices=False):
    norms = input.norm(dim=1, keepdim=True, p=2)
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
    if not torch.all(geoopt.utils.canonical_manifold(ball).k.le(0)):
        raise NotImplementedError("Not implemented for positive curvature")
    gamma = ball.lambda_x(input, dim=-3, keepdim=True)
    numerator = torch.nn.functional.avg_pool2d(
        input * gamma,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=False,
    )
    denominator = torch.nn.functional.avg_pool2d(
        gamma - 1,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=False,
    )
    output = numerator / denominator
    alpha = torch.tensor(0.5, dtype=input.dtype, device=input.device)
    output = ball.mobius_scalar_mul(alpha, output, dim=-3)
    return output


def mobius_adaptive_avg_pool2d(input, output_size, *, ball):
    if not torch.all(geoopt.utils.canonical_manifold(ball).k.le(0)):
        raise NotImplementedError("Not implemented for positive curvature")
    gamma = ball.lambda_x(input, dim=-3, keepdim=True)
    numerator = torch.nn.functional.adaptive_avg_pool2d(
        input * gamma, output_size=output_size
    )
    denominator = torch.nn.functional.adaptive_avg_pool2d(
        gamma - 1, output_size=output_size
    )
    output = numerator / denominator
    alpha = torch.tensor(0.5, dtype=input.dtype, device=input.device)
    output = ball.mobius_scalar_mul(alpha, output, dim=-3)
    return output


def mobius_batch_norm_nd(
    input,
    running_midpoint,
    running_variance,
    beta1,
    beta2,
    alpha=None,
    bias=None,
    epsilon=1e-4,
    training=True,
    *,
    ball,
    n,
):
    dim = -n - 1
    if alpha is None:
        alpha = torch.tensor(1.0, dtype=input.dtype, device=input.device)
    else:
        alpha = alpha.view(alpha.shape + (1,) * n)
    if training:
        reduce_dim = tuple(range(-n, 0)) + tuple(
            range(-input.dim(), -running_midpoint.dim() - n)
        )
        midpoint = ball.weighted_midpoint(
            input, dim=dim, reducedim=reduce_dim, keepdim=True
        )
        variance = ball.dist2(midpoint, input, dim=dim, keepdim=True)
        variance = variance.mean(dim=reduce_dim, keepdim=True)
        input = ball.mobius_add(-midpoint, input, dim=dim)
        input = ball.mobius_scalar_mul(
            alpha / (variance + epsilon) ** 0.5, input, dim=dim
        )
        with torch.no_grad():
            running_variance.lerp_(variance.view_as(running_variance), beta1)
            beta2 = torch.as_tensor(beta2, dtype=input.dtype, device=input.device)
            running_midpoint.set_(
                ball.geodesic(
                    beta2, midpoint.view_as(running_midpoint), running_midpoint, dim=-1
                )
            )
    else:
        running_midpoint = running_midpoint.view(running_midpoint.shape + (1,) * n)
        running_variance = running_variance.view(running_variance.shape + (1,) * n)
        input = ball.mobius_add(-running_midpoint, input, dim=dim)
        input = ball.mobius_scalar_mul(
            alpha / (running_variance + epsilon) ** 0.5, input, dim=dim
        )
    if bias is not None:
        bias = bias.view(bias.shape + (1,) * n)
        input = ball.mobius_add(input, bias, dim=dim)
    return input


mobius_batch_norm = functools.partial(mobius_batch_norm_nd, n=0)
mobius_batch_norm1d = functools.partial(mobius_batch_norm_nd, n=1)
mobius_batch_norm2d = functools.partial(mobius_batch_norm_nd, n=2)


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
        new_tangent = tangent @ weight.t()
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
            new_tangent = tangent @ weight.t()
            output = ball_out.expmap0(new_tangent)
            if bias is not None:
                output = ball_out.mobius_add(output, bias)
    return output


def mobius_conv2d(
    input,
    weight_mm,
    weight_avg,
    stride=1,
    padding=0,
    dilation=1,
    points_in=1,
    points_out=1,
    *,
    ball,
    ball_out=None,
):
    # input BxPxDxWxH
    assert weight_avg.shape[:2] == (points_out, points_in)
    if ball_out is None:
        ball_out = ball
    if not torch.all(geoopt.utils.canonical_manifold(ball).k.le(0)):
        raise NotImplementedError("Not implemented for positive curvature")
    if weight_mm is not None:
        input = ball.logmap0(input, dim=2)
        input = input.transpose(1, 2).reshape(input.shape[0], -1, *input.shape[-2:])
        input = torch.nn.functional.conv2d(
            input, weight_mm.view(*weight_mm.shape, 1, 1)
        )
        in_shape = input.shape
        input = input.view(input.shape[0], -1, points_in, *input.shape[-2:])
        out_dim = input.shape[1]
        input = ball_out.expmap0(input, dim=1)
    else:
        input = input.transpose(1, 2)
        in_shape = input.shape[0], -1, *input.shape[-2:]
        out_dim = input.shape[1]
    gamma = ball_out.lambda_x(input, dim=1, keepdim=True)
    nominator = (input * gamma).reshape(in_shape)
    denominator = (gamma - 1).view(input.shape[0], points_in, *input.shape[-2:])
    # [B, (p1_0, p2_0, p3_0, p4_0, ..., p1_D, p2_0, p3_D, p4_D), H, W)
    weight_avg_d = weight_avg.repeat_interleave(out_dim, dim=0)
    output_nominator = torch.nn.functional.conv2d(
        nominator,
        weight_avg_d,
        groups=out_dim,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    output_denominator = torch.nn.functional.conv2d(
        denominator, weight_avg.abs(), stride=stride, padding=padding, dilation=dilation
    )
    output_denominator = output_denominator.repeat_interleave(out_dim, dim=1)
    two_mean = output_nominator / output_denominator
    two_mean = two_mean.view(two_mean.shape[0], -1, points_out, *two_mean.shape[-2:])
    alpha = torch.tensor(0.5, device=input.device, dtype=input.dtype)
    mean = ball.mobius_scalar_mul(alpha, two_mean, dim=1)
    return mean.transpose(1, 2)  # output BxPxDxWxH
