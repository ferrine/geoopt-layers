import torch


def idx2sign(idx, dim, neg=True):
    """
    Unify idx to be negative or positive, that helps in cases of broadcasting.

    Parameters
    ----------
    idx : int
        current index
    dim : int
        maximum dimension
    neg : bool
        indicate we need negative index

    Returns
    -------
    int
    """
    if neg:
        if idx < 0:
            return idx
        else:
            return (idx + 1) % -(dim + 1)
    else:
        return idx % dim


def poincare2klein(x, *, c=1.0, dim=-1):
    """
    Transform coordinates from Poincare to Klein.

    Parameters
    ----------
    x : tensor
        point on Poincare ball
    c : float
        negative curvature
    dim : int
        dimension to calculate conformal and Lorenz factors

    Returns
    -------
    tensor
        point on Klein disk
    """
    denom = 1 + c * x.pow(2).sum(dim=dim, keepdim=True)
    return 2 * x / denom


def klein2poincare(x, *, c=1.0, dim=-1):
    """
    Transform coordinates from Klein to Poincare.

    Parameters
    ----------
    x : tensor
        point on Klein disk
    c : float
        negative curvature
    dim : int
        dimension to calculate conformal and Lorenz factors

    Returns
    -------
    tensor
        point on Poincare ball
    """
    denom = 1 + torch.sqrt(1 - c * x.pow(2).sum(dim=dim, keepdim=True))
    return x / denom


def lorentz_factor(x: torch.Tensor, *, c=1.0, dim=-1, keepdim=False):
    """
    Calculate Lorenz factor in Klein coordinates.

    Parameters
    ----------
    x : tensor
        point on Klein disk
    c : float
        negative curvature
    dim : int
        dimension to calculate Lorenz factor
    keepdim : bool
        retain the last dim? (default: false)

    Returns
    -------
    tensor
        Lorenz factor
    """
    return 1 / torch.sqrt(1 - c * x.pow(2).sum(dim=dim, keepdim=keepdim))


def _drop_dims(tensor, dims):
    # Workaround to drop several dims in :func:`torch.squeeze`.
    dims = _canonical_dims(dims, tensor.dim())
    slc = tuple(slice(None) if d not in dims else 0 for d in range(tensor.dim()))
    return tensor[slc]


def _canonical_dims(dims, maxdim):
    return tuple(idx2sign(idx, maxdim, neg=False) for idx in dims)


def poincare_mean(xs, weights=None, *, reducedim=None, dim=-1, c=1.0, keepdim=False):
    """
    Compute Einstein midpoint in Poincare coordinates.

    Parameters
    ----------
    xs : tensor
        points on poincare ball
    weights : tensor
        weights for averaging
    reducedim : int|list|tuple
        average dimension
    dim : int
        dimension to calculate conformal and Lorenz factors
    c : float
        ball negative curvature
    keepdim : bool
        retain the last dim? (default: false)

    Returns
    -------
    tensor
        Einstein midpoint in poincare coordinates
    """
    xs = poincare2klein(xs, c=c, dim=dim)
    # klein model
    if reducedim is None:
        reducedim = list(range(xs.dim()))
        del reducedim[dim]
    elif not isinstance(reducedim, (list, tuple)):
        reducedim = (reducedim,)
    reducedim = tuple(reducedim)
    mean = klein_mean(
        xs, weights=weights, reducedim=reducedim, dim=dim, c=c, keepdim=True
    )
    mean = klein2poincare(mean, c=c, dim=dim)
    if not keepdim:
        mean = _drop_dims(mean, reducedim)
    return mean


def klein_mean(xs, weights=None, *, reducedim=None, dim=-1, c=1.0, keepdim=False):
    """
    Compute Einstein Midpoint in Klein coordinates.

    Parameters
    ----------
    xs : tensor
        points on Klein disk
    weights : tensor
        weights for averaging
    reducedim : int|list[int]|tuple[int]
        average dimension
    dim : int
        dimension to calculate conformal and Lorenz factors
    c : float
        ball negative curvature
    keepdim : bool
        retain the last dim? (default: false)

    Returns
    -------
    tensor
        Einstein midpoint in klein coordinates
    """
    lamb = lorentz_factor(xs, c=c, dim=dim, keepdim=True)
    if weights is not None:
        lamb = lamb * weights
    if reducedim is None:
        reducedim = list(range(xs.dim()))
        del reducedim[dim]
    elif not isinstance(reducedim, (list, tuple)):
        reducedim = (reducedim,)
    reducedim = tuple(reducedim)
    mean = torch.sum((lamb * xs), dim=reducedim, keepdim=keepdim) / torch.sum(
        lamb, keepdim=keepdim, dim=reducedim
    ).clamp_min(1e-10)
    # back to poincare
    return mean


def apply_radial(fn, input, basis, *, dim):
    """
    Apply a given function along basis vector provided.

    Parameters
    ----------
    fn : callable
    input : tensor
    basis : tensor
    dim : int

    Returns
    -------
    tensor
    """
    coef = (input * basis).sum(dim=dim, keepdim=True)
    result = fn(coef)
    return input + (result - coef) * basis
