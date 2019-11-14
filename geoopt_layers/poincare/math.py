import functools
import operator
from geoopt.utils import canonical_manifold


__all__ = ["poincare_mean", "poincare_lincomb", "apply_radial"]


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


def poincare_mean(xs, weights=None, *, ball, reducedim=None, dim=-1, keepdim=False):
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
    ball : geoopt.Manifold
        Poincare Ball
    keepdim : bool
        retain the last dim? (default: false)

    Returns
    -------
    tensor
        Einstein midpoint in poincare coordinates
    """
    reducedim = _reduce_dim(xs.dim(), reducedim, dim)
    gamma = ball.lambda_x(xs, dim=dim, keepdim=keepdim)
    if weights is None:
        weights = 1.0
    else:
        weights = weights.unsqueeze(dim)
    two_mean = (
        gamma * weights * xs / ((gamma - 1) * weights).sum(reducedim, keepdim=True)
    )
    mean = ball.mobius_scalar_mul(0.5, two_mean.sum(reducedim, keepdim=True))
    # klein model
    if not keepdim:
        mean = _drop_dims(mean, reducedim)
    return mean


def poincare_lincomb_einstein(
    xs, weights=None, *, ball, reducedim=None, dim=-1, keepdim=False
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
    ball : geoopt.Manifold
        Poincare Ball
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
        xs=xs, weights=weights, reducedim=reducedim, dim=dim, ball=ball, keepdim=True
    )
    if weights is None:
        alpha = functools.reduce(operator.mul, (xs.shape[i] for i in reducedim), 1)
    else:
        alpha = weights.unsqueeze(dim).sum(reducedim, keepdim=True)
    point = ball.mobius_scalar_mul(alpha, midpoint, dim=dim)
    if not keepdim:
        point = _drop_dims(point, reducedim)
    return point


def poincare_lincomb_tangent(
    xs, weights=None, *, ball, reducedim=None, dim=-1, keepdim=False, origin=None
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
    ball : geoopt.Manifold
        Poincare Ball
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
        log_xs = ball.logmap0(xs, dim=dim)
    else:
        log_xs = ball.logmap(xs, origin, dim=dim)
    if weights is not None:
        log_xs = weights.unsqueeze(dim) * log_xs
    reduced = log_xs.sum(reducedim, keepdim=True)
    if origin is None:
        ys = ball.expmap0(reduced, ball, dim=dim)
    else:
        ys = ball.expmap(reduced, origin, dim=dim)
    if not keepdim:
        ys = _drop_dims(ys, reducedim)
    return ys


def poincare_lincomb(
    xs,
    weights=None,
    *,
    ball,
    reducedim=None,
    dim=-1,
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
    ball : geoopt.Manifold
        Poincare Ball
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
            xs=xs,
            weights=weights,
            reducedim=reducedim,
            dim=dim,
            ball=ball,
            keepdim=keepdim,
        )
    else:
        return poincare_lincomb_tangent(
            xs=xs,
            weights=weights,
            reducedim=reducedim,
            dim=dim,
            ball=ball,
            keepdim=keepdim,
            origin=origin,
        )


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
