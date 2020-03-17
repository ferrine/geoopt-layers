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
    ):
        super().__init__()
        self.signed = signed
        self.squared = squared
        self.ball = ball
        self.plane_shape = geoopt.utils.size2shape(plane_shape)
        self.num_planes = num_planes
        self.sphere = geoopt.Sphere()
        self.tangents = geoopt.ManifoldParameter(
            torch.empty(num_planes, plane_shape),
            requires_grad=True,
            manifold=self.sphere,
        )
        if scaled:
            self.scale = torch.nn.Parameter(
                torch.empty(num_planes), requires_grad=True,
            )
        self.log_dist0 = torch.nn.Parameter(torch.empty(num_planes), requires_grad=True,)
        self.reset_parameters()

    @property
    def dist0(self):
        return self.log_dist0.exp()

    @property
    def points(self):
        points = -self.dist0.unsqueeze(-1) * self.tangents
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
            scaled=False,
        )
        if self.scale is not None:
            scale = self.scale.view((-1,) + (1,) * self.n)
            distance = distance * scale
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
        torch.nn.init.xavier_normal_(self.tangents)
        if self.scale is not None:
            self.scale.set_(-self.tangents.norm(dim=-1) / 2)
        self.tangents.proj_()
        torch.nn.init.constant_(self.log_dist0, 1)

    @torch.no_grad()
    def set_parameters_from_linear_operator(self, A, b=None):
        if self.scale is not None:
            self.scale[: A.shape[0]].set_(-A.norm(dim=-1) / 2)
        self.tangents[: A.shape[0]] = A
        self.tangents.proj_()
        if b is not None:
            self.bias[: b.shape[0]] = b
        else:
            self.bias.fill_(1e-4)

    def set_parameters_from_linear_layer(self, linear):
        self.set_parameters_from_linear_operator(linear.weight, linear.bias)


class Distance2PoincareHyperplanes1d(Distance2PoincareHyperplanes):
    n = 1


class Distance2PoincareHyperplanes2d(Distance2PoincareHyperplanes):
    n = 2


class Distance2PoincareHyperplanes3d(Distance2PoincareHyperplanes):
    n = 3
