import torch.nn
from .message_passing import HyperbolicMessagePassing
from ..linear import MobiusLinear


class HyperbolicGCNConv(HyperbolicMessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        improved=False,
        cached=False,
        bias=True,
        normalize=True,
        *,
        ball,
        ball_out=None,
        learn_origin=False,
        aggr_method="einstein",
    ):
        if ball_out is None:
            ball_out = ball
        super(HyperbolicGCNConv, self).__init__(
            aggr="mean", ball=ball_out, aggr_method=aggr_method
        )
        self.ball_in = ball
        self.ball_out = ball_out

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize

        self.lin = MobiusLinear(
            in_channels,
            out_channels,
            bias=False,
            ball=self.ball_in,
            ball_out=self.ball_out,
            learn_origin=learn_origin,
        )
        bias_shape = out_channels if bias else None
        self.register_origin(
            "bias",
            self.ball_out,
            origin_shape=bias_shape,
            parameter=True,
            allow_none=True,
        )
        self.cached_result = None
        self.cached_num_edges = None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        self.lin.reset_parameters()
        self.cached_result = None
        self.cached_num_edges = None
        if self.bias is not None:
            self.bias.zero_()

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
        x = self.lin(x)

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
        # slightly modified for re-weightning
        return self.propagate(edge_index, x=x, edge_weight=norm)

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = self.ball_out.mobius_add(aggr_out, self.bias)
        return aggr_out

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, self.in_channels, self.out_channels
        )
