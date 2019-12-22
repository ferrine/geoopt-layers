from typing import Union
from ..base import ManifoldModule
from .math import apply_radial
import torch


__all__ = ["Noise", "Discretization"]


class Noise(ManifoldModule):
    """

    Parameters
    ----------
    alpha : float
        multiplicative part
    beta : float
        additive part
    ball : geoopt.PoincareBall
    dim : int
        dimension to apply
    grad : bool
        allow grads
    backwards : bool
        only push back
    """

    def __init__(
        self, alpha=0.05, beta=0.0, *, ball, dim=-1, grad=False, backwards=True
    ):
        super().__init__()
        self.ball = ball
        self.dim = dim
        self.grad = grad
        self.alpha = alpha
        self.beta = beta
        self.backwards = backwards

    def get_sigma(self, input: torch.Tensor) -> Union[torch.Tensor, float]:
        beta = self.beta
        if self.alpha > 0:
            dist = self.ball.dist0(input, keepdim=True, dim=self.dim)
            alpha = dist * self.alpha
        else:
            alpha = 0.0
        sigma = (alpha ** 2 + beta ** 2) ** 0.5 / input.size(self.dim) ** 0.5
        return sigma

    def forward(self, input: torch.Tensor):
        if self.training and (self.alpha > 0 or self.beta > 0):
            with torch.set_grad_enabled(self.grad):
                std = self.get_sigma(input)
            eps = torch.randn_like(input) * std
            if self.backwards:
                eps = apply_radial(lambda x: x.clamp_max(0), eps, input)
            return self.ball.expmap(input, eps, dim=self.dim)
        else:
            return input


class Discretization(ManifoldModule):
    r"""
    Stochastic discretization module.

    Parameters
    ----------
    granularity : float
        Granularity of the discretization
    p : float
        :math:`p \in [0, 1]` probability of discretization per dim
    ball : geoopt.PoincareBall
        The manifold instance to use
    dim : int
        Manifold dimension

    Notes
    -----
    As granularity goes to :math:`\infty`, we obtain dropout without mean normalization
    """

    def __init__(self, granularity=0.5, p=0.5, *, ball, dim=-1):
        super().__init__()
        assert granularity > 0
        assert 0 <= p <= 1
        self.granularity = granularity
        self.p = p
        self.ball = ball
        self.dim = dim

    def forward(self, input: torch.Tensor):
        if self.p > 0 and self.training:
            log = self.ball.logmap0(input, dim=self.dim)
            with torch.no_grad():
                quantized = (log * self.granularity).round_().div_(self.granularity)
            mask = torch.empty_like(log, dtype=torch.uint8).bernoulli_(p=self.p)
            qlog = torch.where(mask, quantized, log)
            return self.ball.expmap0(qlog, dim=self.dim)
        else:
            return input
