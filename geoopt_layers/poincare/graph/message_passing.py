from ..math import poincare_mean_scatter
from torch_geometric.nn.conv.message_passing import (
    MessagePassing,
    msg_special_args,
    aggr_special_args,
    update_special_args,
)
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
    @property
    def __args__(self):
        msg_args = set(self.__msg_params__.keys()) - msg_special_args
        aggr_args = set(self.__aggr_params__.keys()) - aggr_special_args
        update_args = set(self.__update_params__.keys()) - update_special_args

        return set().union(msg_args, aggr_args, update_args)

    @__args__.setter
    def __args__(self, new):
        pass

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
        super(HyperbolicMessagePassing, self).__init__(
            aggr=aggr, flow=flow, node_dim=node_dim
        )
        self.ball = ball
        self.aggr_method = aggr_method

    def aggregate(self, out, index, dim_size, edge_weight=None):
        out = poincare_mean_scatter(
            out,
            index,
            weights=edge_weight,
            dim=self.node_dim,
            dim_size=dim_size,
            ball=self.ball,
            lincomb=self.aggr == "add",
            method=self.aggr_method,
        )
        return out
