from geoopt_layers.base import ManifoldModule
import torch


__all__ = [
    "GromovProductHyperbolic",
    "GromovProductHyperbolic1d",
    "GromovProductHyperbolic2d",
    "GromovProductHyperbolic3d",
]


class GromovProductHyperbolic(ManifoldModule):
    n = 0

    def __init__(
        self, in_dim: int, num_products: int, *, ball, learn_reference=True, bias=True
    ):
        super().__init__()
        if not isinstance(num_products, int) or num_products < 1:
            raise TypeError("num_products should be int > 0")
        self.num_products = num_products
        self.manifold = ball
        self.log_points = torch.nn.Parameter(
            torch.empty(num_products, in_dim), requires_grad=True
        )
        self.log_reference = torch.nn.Parameter(
            torch.zeros(num_products, in_dim), requires_grad=learn_reference
        )
        if bias:
            self.bias = torch.nn.Parameter(
                torch.empty(num_products), requires_grad=True
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        self.log_points.normal_(std=self.log_points.shape[-1] ** -0.5)
        self.log_reference.fill_(0)
        if self.bias is not None:
            self.bias.fill_(0)

    @property
    def points(self):
        return self.manifold.expmap0(self.log_points)

    @property
    def reference(self):
        return self.manifold.expmap0(self.log_reference)

    def forward(self, input):
        input = input.unsqueeze(-self.n - 1)
        points = self.points.permute(1, 0)
        points = points.view(points.shape + (1,) * self.n)
        reference = self.reference.permute(1, 0)
        reference = reference.view(reference.shape + (1,) * self.n)

        dist_ij = self.manifold.dist2(input, points, dim=-self.n - 2)
        dist_i = self.manifold.dist2(reference, input, dim=-self.n - 2)
        dist_j = self.manifold.dist2(reference, points, dim=-self.n - 2)
        inner = (dist_i + dist_j - dist_ij) / 2
        if self.bias is not None:
            inner += self.bias.view(self.bias.shape + (1,) * self.n)
        return inner


class GromovProductHyperbolic1d(GromovProductHyperbolic):
    n = 1


class GromovProductHyperbolic2d(GromovProductHyperbolic):
    n = 2


class GromovProductHyperbolic3d(GromovProductHyperbolic):
    n = 3
