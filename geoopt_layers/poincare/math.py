from geoopt.utils import size2shape, canonical_manifold
import torch
from geoopt.utils import drop_dims, clamp_abs

__all__ = ["poincare_mean", "poincare_mean", "apply_radial", "poincare_mean_scatter"]


def _reduce_dim(maxdim, reducedim, dim):
    if reducedim is None:
        reducedim = list(range(maxdim))
        del reducedim[dim]
    else:
        reducedim = size2shape(reducedim)
    return reducedim


def poincare_mean_tangent(
    xs,
    weights=None,
    *,
    ball,
    reducedim=None,
    dim=-1,
    keepdim=False,
    origin=None,
    lincomb=False,
):
    """
    Compute linear combination of Poincare points in tangent space [1]_.

    Parameters
    ----------
    xs : tensor
        points on poincare ball
    weights : tensor
        weights for averaging (make sure they broadcast correctly and manifold dimension is skipped)
    reducedim : int|list
        average dimension
    dim : int
        dimension to calculate conformal and Lorenz factors
    ball : geoopt.Manifold
        Poincare Ball
    keepdim : bool
        retain the last dim? (default: false)
    origin : tensor
        origin for logmap (make sure it broadcasts)
    lincomb : bool
        linkomb implementation

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
    if lincomb:
        reduced = log_xs.sum(reducedim, keepdim=True)
    else:
        reduced = log_xs.mean(reducedim, keepdim=True)
    if origin is None:
        ys = ball.expmap0(reduced, ball, dim=dim)
    else:
        ys = ball.expmap(reduced, origin, dim=dim)
    if not keepdim:
        ys = drop_dims(ys, reducedim)
    return ys


def poincare_mean(
    xs,
    weights=None,
    *,
    ball,
    reducedim=None,
    dim=-1,
    keepdim=False,
    method="einstein",
    origin=None,
    lincomb=False,
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
    lincomb : bool
        linkomb implementation

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
        if reducedim is not None and not isinstance(reducedim, (list, tuple)):
            reducedim = (reducedim,)
        return ball.weighted_midpoint(
            xs=xs,
            weights=weights,
            reducedim=reducedim,
            dim=dim,
            keepdim=keepdim,
            lincomb=lincomb,
        )
    else:
        return poincare_mean_tangent(
            xs=xs,
            weights=weights,
            reducedim=reducedim,
            dim=dim,
            ball=ball,
            keepdim=keepdim,
            origin=origin,
            lincomb=lincomb,
        )


def poincare_mean_einstein_scatter(
    src, index, weights=None, dim=0, dim_size=None, *, ball, lincomb=False
):
    """
    Compute Scattered Einstein midpoint in Poincare coordinates.
    """
    import torch_scatter

    gamma = ball.lambda_x(src, keepdim=True)
    nominator = gamma * src
    denominator = gamma - 1
    if weights is not None:
        weights = weights.unsqueeze(-1)
        nominator = nominator * weights
        denominator = denominator * weights
    nominator = torch_scatter.scatter_add(nominator, index, dim, None, dim_size)
    denominator = torch_scatter.scatter_add(denominator, index, dim, None, dim_size)
    two_mean = nominator / clamp_abs(denominator, 1e-15)
    a_mean = ball.mobius_scalar_mul(
        torch.tensor(0.5, dtype=two_mean.dtype, device=two_mean.device), two_mean
    )
    k = canonical_manifold(ball).k
    if torch.any(k.gt(0)):
        # check antipode
        b_mean = ball.antipode(a_mean)
        a_means = torch.index_select(a_mean, index=index, dim=dim)
        b_means = torch.index_select(b_mean, index=index, dim=dim)
        a_dists = ball.dist(a_means, src, keepdim=True)
        b_dists = ball.dist(b_means, src, keepdim=True)
        a_dist = torch_scatter.scatter_add(a_dists, index, dim, None, dim_size)
        b_dist = torch_scatter.scatter_add(b_dists, index, dim, None, dim_size)
        better = k.gt(0) & (b_dist < a_dist)
        a_mean = torch.where(better, b_mean, a_mean)
    if lincomb:
        if weights is None:
            weights = src.new_full((src.size(dim),), 0.5)
        if weights.dim() == 1:
            alpha = torch_scatter.scatter_add(weights, index, 0, None, dim_size)
            shape = [1 for _ in range(two_mean.dim())]
            shape[dim] = -1
            alpha = alpha.view(shape)
        else:
            alpha = torch_scatter.scatter_add(weights, index, dim, None, dim_size) / 2
        a_mean = ball.mobius_scalar_mul(alpha, a_mean)
    return a_mean


def poincare_mean_tangent_scatter(
    src, index, weights=None, dim=0, dim_size=None, *, ball, lincomb=False, origin=None
):
    import torch_scatter

    if origin is None:
        log = ball.logmap0(src)
    else:
        log = ball.logmap(origin, src)

    if weights is None:
        if lincomb:
            # src, index, dim, None, dim_size
            result = torch_scatter.scatter_add(log, index, dim, None, dim_size)
        else:
            result = torch_scatter.scatter_mean(log, index, dim, None, dim_size)
    elif weights.dim() == 1:
        shape = [1 for _ in range(log.dim())]
        shape[dim] = -1
        weights = weights.view(shape)
        result = torch_scatter.scatter_add(weights * log, index, dim, None, dim_size)
        if not lincomb:
            denom = torch_scatter.scatter_add(weights, index, dim, None, dim_size)
            result = result / denom
    else:
        weights = weights.unsqueeze(-1)
        result = torch_scatter.scatter_add(weights * log, index, dim, None, dim_size)
        if not lincomb:
            denom = torch_scatter.scatter_add(weights, index, dim, None, dim_size)
            result = result / denom
    if origin is None:
        exp = ball.expmap0(result)
    else:
        exp = ball.expmap(origin, result)
    return exp


def poincare_mean_scatter(
    src,
    index,
    weights=None,
    dim=0,
    dim_size=None,
    *,
    ball,
    lincomb=False,
    origin=None,
    method="einstein",
):
    assert method in {"einstein", "tangent"}
    if method == "einstein":
        assert origin is None
        return poincare_mean_einstein_scatter(
            src,
            index,
            weights=weights,
            dim=dim,
            dim_size=dim_size,
            ball=ball,
            lincomb=lincomb,
        )
    else:
        return poincare_mean_tangent_scatter(
            src,
            index,
            weights=weights,
            dim=dim,
            dim_size=dim_size,
            ball=ball,
            lincomb=lincomb,
            origin=origin,
        )


def apply_radial(fn, input, basis, *, dim=-1, norm=True):
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
