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
            t = torch.empty(shape, device=input.device, dtype=input.dtype).uniform_(
                1 - self.gamma, 1 + self.gamma
            )
            input = self.ball.mobius_scalar_mul(t, input, dim=self.dim)
        if self.training and (self.alpha > 0 or self.beta > 0):
            with torch.set_grad_enabled(self.grad):
                std = self.get_sigma(input)
            eps = torch.randn_like(input) * std
            if self.backwards:
                # make sure expectation is still zero
                eps = apply_radial(
                    lambda x: std * (2 / 3.14 / input.size(self.dim)) ** 0.5 - x.abs(),
                    eps,
                    input,
                )
            return self.ball.expmap(input, eps, dim=self.dim)
        else:
            return input

    def extra_repr(self) -> str:
        return "alpha={alpha}, beta={beta}, gamma={gamma}, grad={grad}, backwards={backwards}".format(**self.__dict__)


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

    def __init__(self, granularity=0.5, p=0.5, *, ball, dim=-1, grad=False):
        super().__init__()
        assert granularity > 0
        assert 0 <= p <= 1
        self.granularity = granularity
        self.p = p
        self.ball = ball
        self.dim = dim
        self.grad = grad

    def forward(self, input: torch.Tensor):
        if self.p > 0 and self.training:
            log = self.ball.logmap0(input, dim=self.dim)
            qlog = Quantize.apply(log, self.granularity, self.p, self.grad)
            return self.ball.expmap0(qlog, dim=self.dim)
        else:
            return input

    def extra_repr(self) -> str:
        return "granularity={granularity}, p={p}, grad={grad}".format(**self.__dict__)


class Quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, granularity: float, p: float, grad: bool) -> torch.Tensor:
        quantized = (input * granularity).round_().div_(granularity)
        mask = torch.empty_like(input, dtype=torch.uint8).bernoulli_(p=p)
        if not grad:
            ctx.save_for_backward(mask)
        else:
            ctx.save_for_backward(None)
        return torch.where(mask, quantized, input)

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        if mask is None:
            return grad_output, None, None, None
        else:
            return (~mask).mul(grad_output), None, None, None
