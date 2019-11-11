from ..utils import ManifoldModule
from .functional import mobius_batch_norm2d
import torch
import geoopt

__all__ = ["MobiusBatchNorm2d"]


class MobiusBatchNorm2d(ManifoldModule):
    def __init__(
        self, dimension, alpha=True, epsilon=1e-4, beta1=0.9, beta2=0.9, *, ball
    ):
        super().__init__()
        self.ball = ball
        self.register_buffer("running_midpoint", torch.zeros(1, dimension, 1, 1))
        self.register_buffer("running_variance", torch.ones(()))
        if alpha:
            self.register_parameter("alpha", geoopt.ManifoldParameter(torch.ones(())))
        else:
            self.register_parameter("alpha", None)
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2

    def forward(self, input):
        self.ball.assert_attached(input)
        return mobius_batch_norm2d(
            input=input,
            running_midpoint=self.running_midpoint,
            running_variance=self.running_variance,
            alpha=self.alpha,
            training=self.training,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon,
            ball=self.ball,
        )

    def extra_repr(self):
        return "{dimension}, eps={epsilon}, beta1={beta1}, beta2={beta2}, alpha={alpha}, c={c}".format(
            **self.__dict__,
            alpha=self.alpha is not None,
            dimension=self.running_midpoint.size(1),
        )

    @torch.no_grad()
    def reset_parameters(self):
        self.running_midpoint.zero_()
        self.running_variance.ones_()
        if self.alpha is not None:
            self.alpha.ones_()
