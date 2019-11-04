import torch.nn
import geoopt


class MobiusLinear(torch.nn.Linear):
    """
    Hyperbolic Linear Layer.

    The layer performs

    1. Linear transformation in the tangent space of the origin in the input manifold
    2. Depending on the output manifold, performs either parallel translation to the new origin
    3. Does exponential map from the new origin into the new Manifold

    There are some conventions that should be taken in account:

    - If instances of `ball` and `ball_out` are the same, then input and output manifolds are assumed to be the same.
        In this case it is required to perform parallel transport between tangent spaces of origins. In other case, the
        resulting tangent vector (after Linear transformation) is mapped directly to the
        new Manifold without parallel transport.
    - If input and output manifolds are the same, it is required to have same input and output dimension. Please create
        new instance of :class:`PoincareBall` if you want to change the dimension.

    Parameters
    ----------
    in_features : int
        input dimension
    out_features : int
        output dimension
    bias : bool
        add bias?
    ball : geoopt.PoincareBall
        incoming manifold
    ball_out : Optional[geoopt.PoincareBall]
        output manifold
    learn_origin : bool
        add learnable origins for logmap and expmap?
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        *,
        ball: geoopt.PoincareBall,
        ball_out=None,
        learn_origin=False,
    ):
        # for manifolds that have parameters like Poincare Ball
        # we have to attach them to the closure Module.
        # It is hard to implement device allocation for manifolds in other case.
        if ball_out is None or ball_out is ball:
            if not in_features == out_features:
                raise ValueError(
                    "Please explicitly provide a different instance of PoincareBall to create different out_features"
                )
            ball_out = ball
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.ball = ball
        self.ball_out = ball_out
        if self.bias is not None:
            self.bias = geoopt.ManifoldParameter(self.bias, manifold=self.ball_out)
        if learn_origin:
            self.source_origin = geoopt.ManifoldParameter(self.ball.origin(in_features))
            self.target_origin = geoopt.ManifoldParameter(
                self.ball_out.origin(out_features)
            )
        else:
            self.register_buffer("source_origin", None)
            self.register_buffer("target_origin", None)
        self.reset_parameters()

    def forward(self, input):
        self.ball.assert_attached(input)
        return mobius_linear(
            input,
            weight=self.weight,
            bias=self.bias,
            ball=self.ball,
            ball_out=self.ball_out,
            source_origin=self.source_origin,
            target_origin=self.target_origin,
        )

    @torch.no_grad()
    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.zero_()


def mobius_linear(
    input,
    weight,
    bias=None,
    *,
    ball: geoopt.PoincareBall,
    ball_out: geoopt.PoincareBall,
    source_origin=None,
    target_origin=None,
):
    if source_origin is not None and target_origin is not None:
        # We need to take care of origins
        tangent = ball.logmap(source_origin, input)
        new_tangent = tangent @ weight
        if ball is ball_out:
            # In case same manifolds are used, we need to perform parallel transport
            new_tangent = ball.transp(source_origin, target_origin, new_tangent)
        output = ball_out.expmap(target_origin, new_tangent)
        if bias is not None:
            output = ball_out.mobius_add(output, bias)
    else:
        if ball is ball_out:
            output = ball.mobius_matvec(weight, input)
            if bias is not None:
                output = ball.mobius_add(output, bias)
        else:
            tangent = ball.logmap0(input)
            new_tangent = tangent @ weight
            output = ball_out.expmap0(new_tangent)
            if bias is not None:
                output = ball_out.mobius_add(output, bias)
    return output
