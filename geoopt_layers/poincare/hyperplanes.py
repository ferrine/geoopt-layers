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
        std=1.0,
    ):
        super().__init__()
        self.signed = signed
        self.squared = squared
        self.ball = ball
        self.plane_shape = geoopt.utils.size2shape(plane_shape)
        self.num_planes = num_planes
        self.points = geoopt.ManifoldParameter(
            torch.empty(num_planes, plane_shape), manifold=self.ball
        )
        self.tangents = torch.nn.Parameter(
            torch.empty(num_planes, plane_shape), requires_grad=True
        )
        self.scaled = scaled
        self.std = std
        self.reset_parameters()

    def forward(self, input):
        input_p = input.unsqueeze(-self.n - 1)
        points = self.points.view(self.points.shape + (1,) * self.n).transpose(1, 0)
        tangents = self.tangents.view(self.points.shape + (1,) * self.n).transpose(1, 0)

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
        direction = torch.randn_like(self.points)
        direction /= direction.norm(dim=-1, keepdim=True)
        distance = torch.empty_like(self.points[..., :1]).normal_(std=self.std)
        self.points.set_(self.ball.expmap0(direction * distance))
        self.tangents.set_(self.points.clone())

    @staticmethod
    def hyperplane_from_linear_operator(A, b=None):
        tangents = A
        if b is None:
            points = torch.zeros_like(tangents)
        else:
            points = (
                -b.unsqueeze(-1) * tangents / tangents.pow(2).sum(keepdims=True, dim=-1)
            )
        return points, tangents / 2

    @torch.no_grad()
    def set_parameters_from_linear_operator(self, A, b=None):
        assert A.shape == (self.num_planes, self.plane_shape[0])
        points, tangents = self.hyperplane_from_linear_operator(A, b)
        points = self.ball.expmap0(points)
        self.points.set_(points)
        self.tangents.set_(tangents)

    def set_parameters_from_linear_layer(self, linear):
        self.set_parameters_from_linear_operator(linear.weight, linear.bias)


class Distance2PoincareHyperplanes1d(Distance2PoincareHyperplanes):
    n = 1


class Distance2PoincareHyperplanes2d(Distance2PoincareHyperplanes):
    n = 2


class Distance2PoincareHyperplanes3d(Distance2PoincareHyperplanes):
    n = 3
