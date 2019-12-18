from typing import Union
from ..base import ManifoldModule
import torch


class Noise(ManifoldModule):
    def __init__(self, alpha=0.05, beta=0.0, *, ball, dim=-1, grad=True):
        super().__init__()
        self.ball = ball
        self.dim = dim
        self.grad = grad
        self.alpha = alpha
        self.beta = beta

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
            return self.ball.expmap(input, eps, dim=self.dim)
        else:
            return input
