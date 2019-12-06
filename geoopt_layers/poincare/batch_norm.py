from geoopt_layers.base import ManifoldModule
from .functional import mobius_batch_norm_nd
import torch
import geoopt

__all__ = [
    "MobiusBatchNorm",
    "MobiusBatchNorm1d",
    "MobiusBatchNorm2d",
    "MobiusBatchNorm3d",
]


class MobiusBatchNorm(ManifoldModule):
    n = 0

    def __init__(
        self,
        dimension,
        alpha=True,
        bias=False,
        epsilon=1e-4,
        beta1=0.9,
        beta2=0.9,
        *,
        ball,
    ):
        super().__init__()
        self.ball = ball
        dimension = geoopt.utils.size2shape(dimension)
        self.register_buffer("running_midpoint", torch.zeros(dimension))
        self.register_buffer("running_variance", torch.ones(dimension[:-1] + (1,)))
        if alpha:
            self.register_parameter(
                "alpha", geoopt.ManifoldParameter(torch.ones(dimension[:-1] + (1,)))
            )
        else:
            self.register_parameter("alpha", None)
        if bias:
            self.register_parameter(
                "bias", geoopt.ManifoldParameter(torch.zeros(dimension), manifold=ball)
            )
        else:
            self.register_parameter("bias", None)
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2

    def forward(self, input):
        return mobius_batch_norm_nd(
            input=input,
            running_midpoint=self.running_midpoint,
            running_variance=self.running_variance,
            alpha=self.alpha,
            bias=self.bias,
            training=self.training,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon,
            ball=self.ball,
            n=self.n,
        )

    def extra_repr(self):
        return "{dimension}, eps={epsilon}, beta1={beta1}, beta2={beta2}, alpha={alpha}, bias={bias}".format(
            **self.__dict__,
            alpha=self.alpha is not None,
            bias=self.bias is not None,
            dimension=self.running_midpoint.shape[:-2],
        )

    @torch.no_grad()
    def reset_parameters(self):
        self.running_midpoint.zero_()
        self.running_variance.ones_()
        if self.alpha is not None:
            self.alpha.ones_()
        if self.bias is not None:
            self.bias.zero_()


class MobiusBatchNorm1d(MobiusBatchNorm):
    n = 1


class MobiusBatchNorm2d(MobiusBatchNorm):
    n = 2


class MobiusBatchNorm3d(MobiusBatchNorm):
    n = 3
