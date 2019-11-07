import torch


def mobius_adaptive_max_pool(input, output_size, return_indices=False):
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


def mobius_max_pool(
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
