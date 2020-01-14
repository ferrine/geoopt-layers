from .message_passing import HyperbolicMessagePassing
import torch
import collections
import inspect


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
        aggr="sum",
        bias=True,
        *,
        ball,
        ball_out=None,
        local=False,
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
        self.local = local
        if self.local:
            self.__message_params__ = collections.OrderedDict(
                {
                    "x_i": inspect.Parameter(
                        "x_i", inspect.Parameter.POSITIONAL_OR_KEYWORD
                    ),
                    "x_j": inspect.Parameter(
                        "x_j", inspect.Parameter.POSITIONAL_OR_KEYWORD
                    ),
                }
            )  # ["x_i", "x_j"]
        else:
            self.__message_params__ = collections.OrderedDict(
                {
                    "h_j": inspect.Parameter(
                        "h_j", inspect.Parameter.POSITIONAL_OR_KEYWORD
                    )
                }
            )  # ["h_j"]
        self.weight_neighbors = torch.nn.Parameter(
            torch.empty(in_channels, out_channels), requires_grad=True
        )
        self.weight_loop = torch.nn.Parameter(
            torch.empty(in_channels, out_channels), requires_grad=True
        )
        self.register_origin(
            "bias",
            self.ball_out,
            origin_shape=out_channels,
            parameter=True,
            none=not bias,
        )

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_loop)
        torch.nn.init.xavier_uniform_(self.weight_neighbors)
        if self.bias is not None:
            self.bias.zero_()

    def forward(self, x, edge_index, edge_weight=None, size=None):
        """"""
        y = self.ball.logmap0(x)
        y = y @ self.weight_loop
        y = self.ball_out.expmap0(y)
        if self.local:
            return self.propagate(
                edge_index, size=size, y=y, x=x, edge_weight=edge_weight
            )
        else:
            h = self.ball.logmap0(x)
            h = h @ self.weight_neighbors
            h = self.ball_out.expmap0(h)
            return self.propagate(
                edge_index, size=size, y=y, x=x, h=h, edge_weight=edge_weight
            )

    def message(self, x_i=None, x_j=None, h_j=None):
        if self.local:
            h_j = self.ball.logmap(x_i, x_j)
            h_j = h_j @ self.weight_neighbors
            return self.ball_out.expmap0(h_j)
        else:
            return h_j

    def update(self, aggr_out, y):
        aggr_out = self.ball_out.mobius_add(y, aggr_out)
        if self.bias is not None:
            aggr_out = self.ball_out.mobius_add(aggr_out, self.bias)
        return aggr_out

    def extra_repr(self) -> str:
        return "{} -> {}, local={local}, aggr={aggr}, aggr_method={aggr_method}".format(
            self.in_channels, self.out_channels, **self.__dict__
        )
