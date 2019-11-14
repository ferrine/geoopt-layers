import torch
import functools
import operator
import geoopt


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


def _reduce_dim(maxdim, reducedim, dim):
    if reducedim is None:
        reducedim = list(range(maxdim))
        del reducedim[dim]
    elif not isinstance(reducedim, (list, tuple)):
        reducedim = (reducedim,)
    reducedim = tuple(reducedim)
    return reducedim


def poincare_mean(xs, weights=None, *, reducedim=None, dim=-1, c=1.0, keepdim=False):
    """
    Compute Einstein midpoint in Poincare coordinates.

    Parameters
    ----------
    xs : tensor
        points on poincare ball
    weights : tensor
        weights for averaging (make sure they broadcast correctly and manifold dimension is skipped)
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
    reducedim = _reduce_dim(xs.dim(), reducedim, dim)
    mean = klein_mean(
        xs, weights=weights, reducedim=reducedim, dim=dim, c=c, keepdim=True
    )
    mean = klein2poincare(mean, c=c, dim=dim)
    if not keepdim:
        mean = _drop_dims(mean, reducedim)
    return mean


def poincare_lincomb_einstein(
    xs, weights=None, *, reducedim=None, dim=-1, c=1.0, keepdim=False
):
    """
    Compute linear combination of Poincare points basing on Einstein midpoint [1]_.

    Parameters
    ----------
    xs : tensor
        points on poincare ball
    weights : tensor
        weights for averaging (make sure they broadcast correctly and manifold dimension is skipped)
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
        Linear combination in Poincare ball

    References
    ----------
    .. [1] https://openreview.net/pdf?id=BJg73xHtvr
    """
    reducedim = _reduce_dim(xs.dim(), reducedim, dim)
    midpoint = poincare_mean(
        xs=xs, weights=weights, reducedim=reducedim, dim=dim, c=c, keepdim=True
    )
    if weights is None:
        alpha = functools.reduce(operator.mul, (xs.shape[i] for i in reducedim), 1)
    else:
        alpha = weights.unsqueeze(dim).expand_as(xs).sum(reducedim, keepdim=True)
    point = geoopt.manifolds.poincare.math.mobius_scalar_mul(
        alpha, midpoint, c=c, dim=dim
    )
    if not keepdim:
        point = _drop_dims(point, reducedim)
    return point


def poincare_lincomb_tanget(
    xs, weights=None, *, reducedim=None, dim=-1, c=1.0, keepdim=False, origin=None
):
    """
    Compute linear combination of Poincare points in tangent space [1]_.

    Parameters
    ----------
    xs : tensor
        points on poincare ball
    weights : tensor
        weights for averaging (make sure they broadcast correctly and manifold dimension is skipped)
    reducedim : int|list|tuple
        average dimension
    dim : int
        dimension to calculate conformal and Lorenz factors
    c : float
        ball negative curvature
    keepdim : bool
        retain the last dim? (default: false)
    origin : tensor
        origin for logmap (make sure it broadcasts)

    Returns
    -------
    tensor
        Linear combination in Poincare ball

    References
    ----------
    .. [1] https://openreview.net/pdf?id=BJg73xHtvr
    """
    reducedim = _reduce_dim(xs.dim(), reducedim, dim)
    if origin is None:
        log_xs = geoopt.manifolds.poincare.math.logmap0(xs, c=c, dim=dim)
    else:
        log_xs = geoopt.manifolds.poincare.math.logmap(xs, origin, c=c, dim=dim)
    if weights is not None:
        log_xs = weights.unsqueeze(dim) * log_xs
    reduced = log_xs.sum(reducedim)
    if origin is None:
        ys = geoopt.manifolds.poincare.math.expmap0(reduced, c=c, dim=dim)
    else:
        ys = geoopt.manifolds.poincare.math.expmap(reduced, origin, c=c, dim=dim)
    if not keepdim:
        ys = _drop_dims(ys, reducedim)
    return ys


def poincare_lincomb(
    xs,
    weights=None,
    *,
    reducedim=None,
    dim=-1,
    c=1.0,
    keepdim=False,
    method="einstein",
    origin=None
):
    """
    Compute linear combination of Poincare points.

    Parameters
    ----------
    xs : tensor
        points on poincare ball
    weights : tensor
        weights for averaging (make sure they broadcast correctly and manifold dimension is skipped)
    reducedim : int|list|tuple
        average dimension
    dim : int
        dimension to calculate conformal and Lorenz factors
    c : float
        ball negative curvature
    keepdim : bool
        retain the last dim? (default: false)
    method : str
        one of ``{"einstein", "tangent"}``
    origin : tensor
        origin for logmap (make sure it broadcasts), for ``tangent`` method only

    Returns
    -------
    tensor
        Linear combination in Poincare ball

    References
    ----------
    .. [1] https://openreview.net/pdf?id=BJg73xHtvr
    """
    assert method in {"einstein", "tangent"}
    if method == "einstein":
        assert origin is None
        return poincare_lincomb_einstein(
            xs=xs, weights=weights, reducedim=reducedim, dim=dim, c=c, keepdim=keepdim
        )
    else:
        return poincare_lincomb_tanget(
            xs=xs,
            weights=weights,
            reducedim=reducedim,
            dim=dim,
            c=c,
            keepdim=keepdim,
            origin=origin,
        )


def klein_mean(xs, weights=None, *, reducedim=None, dim=-1, c=1.0, keepdim=False):
    """
    Compute Einstein Midpoint in Klein coordinates.

    Parameters
    ----------
    xs : tensor
        points on Klein disk
    weights : tensor
        weights for averaging (make sure they broadcast correctly and manifold dimension is skipped)
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
        lamb = lamb * weights.unsqueeze(dim)
    reducedim = _reduce_dim(xs.dim(), reducedim, dim)
    mean = torch.sum((lamb * xs), dim=reducedim, keepdim=keepdim) / torch.sum(
        lamb, keepdim=keepdim, dim=reducedim
    ).clamp_min(1e-10)
    # back to poincare
    return mean


def apply_radial(fn, input, basis, *, dim, norm=True):
    """
    Apply a given function along basis vector provided.

    Parameters
    ----------
    fn : callable
        function to apply
    input : tensor
        input tensor
    basis : tensor
        basis, would be normed to one if norm is True
    dim : int
        reduce dimension
    norm : bool
        do we need to norm the basis?

    Returns
    -------
    tensor
        result
    """
    if norm:
        basis = basis / basis.norm(dim=dim, keepdim=True).clamp_min(1e-4)
    coef = (input * basis).sum(dim=dim, keepdim=True)
    result = fn(coef)
    return input + (result - coef) * basis
