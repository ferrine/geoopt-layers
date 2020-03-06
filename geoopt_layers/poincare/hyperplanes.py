from geoopt_layers.base import ManifoldModule
import geoopt
import torch


__all__ = [
    "Distance2PoincareHyperplanes",
    "Distance2PoincareHyperplanes1d",
    "Distance2PoincareHyperplanes2d",
    "Distance2PoincareHyperplanes3d",
]


class Distance2PoincareHyperplanes(ManifoldModule):
    n = 0

    def __init__(
        self,
        plane_shape: int,
        num_planes: int,
        signed=True,
        squared=False,
        scaled=True,
        *,
        ball,
        normalize=False,
    ):
        super().__init__()
        self.signed = signed
        self.squared = squared
        self.ball = ball
        self.plane_shape = geoopt.utils.size2shape(plane_shape)
        self.num_planes = num_planes
        self.weight = torch.nn.Parameter(
            torch.empty(num_planes, plane_shape), requires_grad=True,
        )
        self.bias = torch.nn.Parameter(torch.empty(num_planes), requires_grad=True,)
        self.scaled = scaled
        self.normalize = normalize
        self.reset_parameters()

    @property
    def tangents(self):
        return self.weight / 2

    @property
    def points(self):
        weight = self.weight
        bias = self.bias
        points = -bias.unsqueeze(-1) * weight
        if self.normalize:
            points = points / weight.pow(2).sum(keepdims=True, dim=-1).clamp_min_(1e-15)
        return self.ball.expmap0(points)

    def forward(self, input):
        input_p = input.unsqueeze(-self.n - 1)
        points = self.points
        tangents = self.tangents
        points = points.view(points.shape + (1,) * self.n).transpose(1, 0)
        tangents = tangents.view(tangents.shape + (1,) * self.n).transpose(1, 0)

        distance = self.ball.dist2plane(
            x=input_p,
            p=points,
            a=tangents,
            signed=self.signed,
            dim=-self.n - 2,
            scaled=self.scaled,
        )
        if self.squared and self.signed:
            sign = distance.sign()
            distance = distance ** 2 * sign
        elif self.squared:
            distance = distance ** 2
        return distance

    def extra_repr(self):
        return "plane_shape={plane_shape}, num_planes={num_planes}".format(
            **self.__dict__
        )

    @torch.no_grad()
    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weight)
        torch.nn.init.constant_(self.bias, 0.0)

    @torch.no_grad()
    def set_parameters_from_linear_operator(self, A, b=None):
        self.weight[: A.shape[0]] = A
        if b is not None:
            self.bias[: b.shape[0]] = b
        else:
            self.bias.fill_(0)

    def set_parameters_from_linear_layer(self, linear):
        self.set_parameters_from_linear_operator(linear.weight, linear.bias)


class Distance2PoincareHyperplanes1d(Distance2PoincareHyperplanes):
    n = 1


class Distance2PoincareHyperplanes2d(Distance2PoincareHyperplanes):
    n = 2


class Distance2PoincareHyperplanes3d(Distance2PoincareHyperplanes):
    n = 3
