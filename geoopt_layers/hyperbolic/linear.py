import torch.nn
import geoopt


class MobiusLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        *,
        ball: geoopt.PoincareBall,
        ball_out=None
    ):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        # for manifolds that have parameters like Poincare Ball
        # we have to attach them to the closure Module.
        # It is hard to implement device allocation for manifolds in other case.
        self.ball = ball
        if ball_out is None:
            if not in_features == out_features:
                raise ValueError(
                    "Please explicitly provide ball_out if it not the same"
                )
            ball_out = ball
        self.ball_out = ball_out
        if self.bias is not None:
            self.bias = geoopt.ManifoldParameter(self.bias, manifold=self.ball_out)
        self.reset_parameters()

    def forward(self, input):
        self.ball.assert_attached(input)
        return mobius_linear(
            input,
            weight=self.weight,
            bias=self.bias,
            ball=self.ball,
            ball_out=self.ball_out,
        )

    @torch.no_grad()
    def reset_parameters(self):
        if self.bias is not None:
            self.bias.zero_()


def mobius_linear(
    input,
    weight,
    bias=None,
    *,
    ball: geoopt.PoincareBall,
    ball_out: geoopt.PoincareBall
):
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
