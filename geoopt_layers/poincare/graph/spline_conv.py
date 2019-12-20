import warnings
import math
import torch
from torch.nn import Parameter
from .message_passing import HyperbolicMessagePassing
from ...utils import repeat


try:
    from torch_spline_conv import SplineBasis, SplineWeighting
except ImportError:
    SplineBasis = None
    SplineWeighting = None


class HyperbolicSplineConv(HyperbolicMessagePassing):
    r"""The spline-based convolutional operator from the `"SplineCNN: Fast
    Geometric Deep Learning with Continuous B-Spline Kernels"
    <https://arxiv.org/abs/1711.08920>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \frac{1}{|\mathcal{N}(i)|} \sum_{j \in
        \mathcal{N}(i)} \mathbf{x}_j \cdot
        h_{\mathbf{\Theta}}(\mathbf{e}_{i,j}),

    where :math:`h_{\mathbf{\Theta}}` denotes a kernel function defined
    over the weighted B-Spline tensor product basis.

    .. note::

        Pseudo-coordinates must lay in the fixed interval :math:`[0, 1]` for
        this method to work as intended.

    Parameters
    ----------
    in_channels : int
        Size of each input sample.
    out_channels : int
        Size of each output sample.
    dim : int
        Pseudo-coordinate dimensionality.
    kernel_size : int
        Size of the convolving kernel.
    is_open_spline : bool
        If set to :obj:`False`, the
        operator will use a closed B-spline basis in this dimension.
        (default :obj:`True`)
    degree (int, optional): B-spline basis degrees. (default: :obj:`1`)
    aggr : str
        The aggregation operator to use
        (:obj:`"sum"`, :obj:`"mean"`, default: :obj:`"mean"`).
    root_weight : bool
        If set to :obj:`False`, the layer will
        not add transformed root node features to the output.
        (default: :obj:`True`)
    bias : bool
        If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)
    **kwargs (optional): Additional arguments of
        :class:`geoopt_layers.poincare.graph.HyperbolicMessagePassing`.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        dim,
        kernel_size,
        is_open_spline=True,
        degree=1,
        aggr="mean",
        root_weight=True,
        bias=True,
        *,
        ball,
        ball_out=None,
        aggr_method="einstein",
    ):
        if ball_out is None:
            ball_out = ball
        super(HyperbolicSplineConv, self).__init__(
            aggr=aggr, ball=ball_out, aggr_method=aggr_method
        )
        self.ball_in = ball
        self.ball_out = ball_out
        if SplineBasis is None:
            raise ImportError("`HyperbolicSplineConv` requires `torch-spline-conv`.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.degree = degree

        kernel_size = torch.tensor(repeat(kernel_size, dim), dtype=torch.long)
        self.register_buffer("kernel_size", kernel_size)

        is_open_spline = repeat(is_open_spline, dim)
        is_open_spline = torch.tensor(is_open_spline, dtype=torch.uint8)
        self.register_buffer("is_open_spline", is_open_spline)

        K = kernel_size.prod().item()
        self.weight = Parameter(
            torch.empty(K, in_channels, out_channels), requires_grad=True
        )

        if root_weight:
            self.root = Parameter(
                torch.empty(in_channels, out_channels), requires_grad=True
            )
        else:
            self.register_parameter("root", None)

        bias_shape = out_channels if bias else None
        self.register_origin(
            "bias",
            self.ball_out,
            origin_shape=bias_shape,
            parameter=True,
            allow_none=True,
        )

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        size = self.in_channels * self.weight.size(0)
        bound = 1.0 / math.sqrt(size)
        self.weight.uniform_(-bound, bound)
        if self.root is not None:
            self.root.uniform_(-bound, bound)
        if self.bias is not None:
            self.bias.zero_()

    def forward(self, x, edge_index, pseudo, edge_weight=None):
        """"""
        try:
            import torch_geometric
        except ImportError:  # for debug mode only
            torch_geometric = None
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = pseudo.unsqueeze(-1) if pseudo.dim() == 1 else pseudo

        if not x.is_cuda:
            warnings.warn(
                "We do not recommend using the non-optimized CPU "
                "version of SplineConv. If possible, please convert "
                "your data to the GPU."
            )

        if torch_geometric is not None and torch_geometric.is_debug_enabled():
            if x.size(1) != self.in_channels:
                raise RuntimeError(
                    "Expected {} node features, but found {}".format(
                        self.in_channels, x.size(1)
                    )
                )

            if pseudo.size(1) != self.dim:
                raise RuntimeError(
                    (
                        "Expected pseudo-coordinate dimensionality of {}, but "
                        "found {}"
                    ).format(self.dim, pseudo.size(1))
                )

            min_index, max_index = edge_index.min(), edge_index.max()
            if min_index < 0 or max_index > x.size(0) - 1:
                raise RuntimeError(
                    (
                        "Edge indices must lay in the interval [0, {}]"
                        " but found them in the interval [{}, {}]"
                    ).format(x.size(0) - 1, min_index, max_index)
                )

            min_pseudo, max_pseudo = pseudo.min(), pseudo.max()
            if min_pseudo < 0 or max_pseudo > 1:
                raise RuntimeError(
                    (
                        "Pseudo-coordinates must lay in the fixed interval [0, 1]"
                        " but found them in the interval [{}, {}]"
                    ).format(min_pseudo, max_pseudo)
                )
        x = self.ball_in.logmap0(x)
        return self.propagate(edge_index, x=x, pseudo=pseudo, edge_weight=edge_weight)

    def message(self, x_j, pseudo):
        data = SplineBasis.apply(
            pseudo,
            self._buffers["kernel_size"],
            self._buffers["is_open_spline"],
            self.degree,
        )
        z_j = SplineWeighting.apply(x_j, self.weight, *data)
        return self.ball_out.expmap0(z_j)

    def update(self, aggr_out, x):
        if self.root is not None:
            z = self.ball_out.expmap0(torch.mm(x, self.root))
            aggr_out = self.ball_out.mobius_add(z, aggr_out)
        if self.bias is not None:
            aggr_out = self.ball_out.mobius_add(aggr_out, self.bias)
        return aggr_out

    def __repr__(self):
        return "{}({}, {}, dim={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.dim
        )
