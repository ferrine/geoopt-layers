from geoopt_layers.utils import idx2sign, prod
from geoopt.utils import size2shape
import torch_scatter
__all__ = ["poincare_mean", "poincare_lincomb", "apply_radial", "poincare_mean_scatter_"]


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
    else:
        reducedim = size2shape(reducedim)
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
    gamma = ball.lambda_x(xs, dim=dim, keepdim=True)
    if weights is None:
        weights = 1.0
    else:
        weights = weights.unsqueeze(dim)
    two_mean = (
        gamma * weights * xs / ((gamma - 1) * weights).sum(reducedim, keepdim=True)
    ).sum(reducedim, keepdim=True)
    mean = ball.mobius_scalar_mul(0.5, two_mean)
    if not keepdim:
        mean = _drop_dims(mean, reducedim)
    return mean


def poincare_mean_scatter_(src, index, dim=0, dim_size=None, *, ball):
    """
    Compute Scattered Einstein midpoint in Poincare coordinates.
    """
    # scatter_*(src, index, dim, None, dim_size, fill_value)
    raise NotImplementedError


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
        alpha = prod((xs.shape[i] for i in reducedim))
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
