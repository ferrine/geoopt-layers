from .. import Distance2PoincareHyperplanes, WeightedPoincareCentroids
import torch_geometric.nn.conv
import torch.nn


class HyperbolicGCNConv(torch_geometric.nn.conv.MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        improved=False,
        cached=False,
        normalize=True,
        num_basis=None,
        num_planes=None,
        balls,
        balls_out=None,
        local=False,
        nonlinearity=torch.nn.Identity(),
    ):
        if not isinstance(balls, list):
            balls = [balls]
        if balls_out is None:
            balls_out = balls
        if not isinstance(balls_out, list):
            balls_out = [balls_out]
        super(HyperbolicGCNConv, self).__init__(aggr="add")
        if num_planes is None:
            num_planes = in_channels
        if num_basis is None:
            num_basis = out_channels
        self.num_basis = num_basis
        self.num_planes = num_planes
        self.balls_in = torch.nn.ModuleList(balls)
        self.balls_out = torch.nn.ModuleList(balls_out)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.local = local
        self.improved = improved
        self.cached = cached
        self.normalize = normalize
        self.hyperplanes = torch.nn.ModuleList(
            [
                Distance2PoincareHyperplanes(
                    in_channels, num_planes, ball=ball, scaled=True, squared=False
                )
                for ball in self.balls_in
            ]
        )
        self.basis = torch.nn.ModuleList(
            [
                WeightedPoincareCentroids(
                    out_channels, num_basis, ball=ball_out, lincomb=True
                )
                for ball_out in self.balls_out
            ]
        )
        self.mixing = torch.nn.Linear(
            num_planes * len(self.balls_in), num_basis * len(self.balls_out)
        )
        self.nonlinearity = nonlinearity
        self.cached_result = None
        self.cached_num_edges = None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        [plane.reset_parameters() for plane in self.hyperplanes]
        [basis.reset_parameters_identity() for basis in self.basis]
        self.mixing.reset_parameters()
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
        from torch_scatter import scatter_add
        from torch_geometric.utils import add_remaining_self_loops

        if edge_weight is None:
            edge_weight = torch.ones(
                (edge_index.size(1),), dtype=dtype, device=edge_index.device
            )

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes
        )

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    "Cached {} number of edges, but found {}. Please "
                    "disable the caching behavior of this layer by removing "
                    "the `cached=True` argument in its constructor.".format(
                        self.cached_num_edges, edge_index.size(1)
                    )
                )

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if self.normalize:
                edge_index, norm = self.norm(
                    edge_index,
                    x.size(self.node_dim),
                    edge_weight,
                    self.improved,
                    x.dtype,
                )
            else:
                norm = edge_weight
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        xs = x.chunk(len(self.balls_in), -1)
        dists = [plane(x) for x, plane in zip(xs, self.hyperplanes)]
        dists = torch.cat(dists, dim=-1)
        return self.propagate(edge_index, dists=dists, norm=norm)

    def message(self, dists_j=None, norm=None):
        return norm.view(-1, 1) * dists_j if norm is not None else dists_j

    def update(self, aggr_out):
        activations = self.nonlinearity(aggr_out)
        activations = self.mixing(activations)
        activations = activations.chunk(len(self.balls_out), -1)
        points = [basis(a) for basis, a in zip(self.basis, activations)]
        return torch.cat(points, dim=-1)

    def extra_repr(self) -> str:
        return "{} -> {}, aggr={aggr}".format(
            self.in_channels, self.out_channels, **self.__dict__
        )

    @torch.no_grad()
    def set_parameters_from_gcn_conv(self, gcn_conv: torch_geometric.nn.conv.GCNConv):
        self.reset_parameters()
        self.hyperplanes.set_parameters_from_linear_operator(gcn_conv.weight.t())
        if gcn_conv.bias is not None:
            self.bias[: gcn_conv.bias.shape[0]] = gcn_conv.bias.clone()

    @classmethod
    def from_gcn_conv(
        cls,
        gcn_conv: torch_geometric.nn.conv.GCNConv,
        nonlinearity=torch.nn.Identity(),
        *,
        ball,
        ball_out=None,
        num_basis=None,
    ):
        layer = cls(
            gcn_conv.in_channels,
            gcn_conv.out_channels,
            num_basis=num_basis,
            nonlinearity=nonlinearity,
            ball=ball,
            ball_out=ball_out,
            improved=gcn_conv.improved,
            normalize=gcn_conv.normalize,
            cached=gcn_conv.cached,
        )
        layer.set_parameters_from_gcn_conv(gcn_conv)
        return layer
