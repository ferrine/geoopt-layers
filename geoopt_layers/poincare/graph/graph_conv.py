from .message_passing import HyperbolicMessagePassing
from ..linear import MobiusLinear


class HyperbolicGraphConv(HyperbolicMessagePassing):
    r"""The graph neural network operator from the `"Weisfeiler and Leman Go
    Neural: Higher-order Graph Neural Networks"
    <https://arxiv.org/abs/1810.02244>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_1 \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \mathbf{\Theta}_2 \mathbf{x}_j.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        aggr="add",
        bias=True,
        *,
        ball,
        ball_out=None,
        learn_origin=False,
        aggr_method="einstein",
    ):
        if ball_out is None:
            ball_out = ball
        super(HyperbolicGraphConv, self).__init__(
            aggr=aggr, ball=ball_out, aggr_method=aggr_method
        )
        self.ball_in = ball
        self.ball_out = ball_out
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin_neighbors = MobiusLinear(
            in_channels,
            out_channels,
            bias=False,
            ball=self.ball_in,
            ball_out=self.ball_out,
            learn_origin=learn_origin,
        )
        self.lin_loop = MobiusLinear(
            in_channels,
            out_channels,
            bias=bias,
            ball=self.ball_in,
            ball_out=self.ball_out,
            learn_origin=learn_origin,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_loop.reset_parameters()
        self.lin_neighbors.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, size=None):
        """"""
        h = self.lin_neighbors(x)
        return self.propagate(edge_index, size=size, x=x, h=h, edge_weight=edge_weight)

    def message(self, h_j, edge_weight):
        return h_j

    def update(self, aggr_out, x):
        return self.ball.mobius_add(aggr_out, self.lin_loop(x))

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, self.in_channels, self.out_channels
        )
