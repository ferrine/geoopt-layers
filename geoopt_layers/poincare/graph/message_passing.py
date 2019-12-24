from ..math import poincare_mean_scatter
from torch_geometric.nn.conv.message_passing import MessagePassing
from ...base import ManifoldModule

special_args = [
    "edge_index",
    "edge_index_i",
    "edge_index_j",
    "size",
    "size_i",
    "size_j",
]
__size_error_msg__ = (
    "All tensors which should get mapped to the same source "
    "or target nodes must be of same size in dimension 0."
)


class HyperbolicMessagePassing(MessagePassing, ManifoldModule):
    __aggr__ = {"sum", "mean"}

    def __init__(
        self,
        aggr="mean",
        flow="source_to_target",
        *,
        ball,
        aggr_method="einstein",
        node_dim=0,
    ):
        assert node_dim in {0}, "other dims are not yet supported"
        super(HyperbolicMessagePassing, self).__init__(aggr=aggr, flow=flow, node_dim=node_dim)
        self.ball = ball
        self.aggr_method = aggr_method

    def propagate(self, edge_index, size=None, edge_weight=None, **kwargs):
        return super().propagate(edge_index, size=size, edge_weight=edge_weight, **kwargs)

    def aggregate(self, out, index, dim=0, dim_size=None, edge_weight=None):
        out = poincare_mean_scatter(
            out,
            index,
            weights=edge_weight,
            dim=dim,
            dim_size=dim_size,
            ball=self.ball,
            lincomb=self.aggr == "sum",
            method=self.aggr_method,
        )
        return out
