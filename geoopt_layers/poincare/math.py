from geoopt.utils import size2shape
import torch
from geoopt_layers.utils import idx2sign, prod

__all__ = ["poincare_mean", "poincare_mean", "apply_radial", "poincare_mean_scatter"]


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


def poincare_mean_einstein(
    xs, weights=None, *, ball, reducedim=None, dim=-1, keepdim=False, lincomb=False
):
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
    lincomb : bool
        linear combination implementation

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
    nominator = (gamma * weights * xs).sum(reducedim, keepdim=True)
    denominator = ((gamma - 1) * weights).sum(reducedim, keepdim=True)
    two_mean = nominator / denominator
    two_mean = torch.where(torch.isfinite(two_mean), two_mean, two_mean.new_zeros(()))
    if lincomb and isinstance(weights, float):
        alpha = 0.5 * prod((xs.shape[i] for i in reducedim))
    elif lincomb:
        alpha = 0.5 * weights.sum(reducedim, keepdim=True)
    else:
        alpha = 0.5
    mean = ball.mobius_scalar_mul(alpha, two_mean, dim=dim)
    if not keepdim:
        mean = _drop_dims(mean, reducedim)
    return mean


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
        ys = _drop_dims(ys, reducedim)
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
        return poincare_mean_einstein(
            xs=xs,
            weights=weights,
            reducedim=reducedim,
            dim=dim,
            ball=ball,
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
    nominator = torch_scatter.scatter_add(nominator, index, dim, None, dim_size, 0)
    denominator = torch_scatter.scatter_add(
        denominator, index, dim, None, dim_size, 1e-5
    )
    two_mean = nominator / denominator
    two_mean = torch.where(torch.isfinite(two_mean), two_mean, two_mean.new_zeros(()))
    if lincomb:
        if weights is None:
            weights = src.new_full((src.size(dim),), 0.5)
        if weights.dim() == 1:
            alpha = torch_scatter.scatter_add(weights, index, 0, None, dim_size, 0)
            shape = [1 for _ in range(two_mean.dim())]
            shape[dim] = -1
            alpha = alpha.view(shape)
        else:
            alpha = (
                torch_scatter.scatter_add(weights, index, dim, None, dim_size, 0) / 2
            )
    else:
        alpha = 0.5
    return ball.mobius_scalar_mul(alpha, two_mean)


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
            result = torch_scatter.scatter_add(log, index, dim, None, dim_size, 0)
        else:
            result = torch_scatter.scatter_mean(log, index, dim, None, dim_size, 0)
    elif weights.dim() == 1:
        shape = [1 for _ in range(log.dim())]
        shape[dim] = -1
        weights = weights.view(shape)
        result = torch_scatter.scatter_add(weights * log, index, dim, None, dim_size, 0)
        if not lincomb:
            denom = torch_scatter.scatter_add(weights, index, dim, None, dim_size, 1e-5)
            result = result / denom
    else:
        weights = weights.unsqueeze(-1)
        result = torch_scatter.scatter_add(weights * log, index, dim, None, dim_size, 0)
        if not lincomb:
            denom = torch_scatter.scatter_add(weights, index, dim, None, dim_size, 1e-5)
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
