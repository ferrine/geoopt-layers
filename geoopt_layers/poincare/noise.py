from typing import Union
from ..base import ManifoldModule
from .math import apply_radial
import torch


class Noise(ManifoldModule):
    """

    Parameters
    ----------
    alpha : float
        multiplicative part
    beta : float
        additive part
    gamma : float
        scale part
    ball : geoopt.PoincareBall
    dim : int
        dimension to apply
    grad : bool
        allow grads
    backwards : bool
        only push back
    """

    def __init__(
        self, alpha=0.05, beta=0.0, gamma=0, *, ball, dim=-1, grad=True, backwards=True
    ):
        super().__init__()
        self.ball = ball
        self.dim = dim
        self.grad = grad
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
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
        if self.training and self.gamma > 0:
            shape = list(input.shape)
            shape[self.dim] = 1
            t = torch.empty(shape).uniform_(1 - self.gamma, 1 + self.gamma)
            input = self.ball.mobius_scalar_mul(t, input, dim=self.dim)
        if self.training and (self.alpha > 0 or self.beta > 0):
            with torch.set_grad_enabled(self.grad):
                std = self.get_sigma(input)
            eps = torch.randn_like(input) * std
            if self.backwards:
                # make sure expectation is still zero
                eps = apply_radial(
                    lambda x: std * (2 / 3.14) ** 0.5 - x.abs(), eps, input
                )
            return self.ball.expmap(input, eps, dim=self.dim)
        else:
            return input
