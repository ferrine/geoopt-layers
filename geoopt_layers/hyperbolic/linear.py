import torch.nn
import geoopt


class MobiusLinear(torch.nn.Linear):
    def __init__(self, *args, ball=None, **kwargs):
        super().__init__(*args, **kwargs)
        # for manifolds that have parameters like Poincare Ball
        # we have to attach them to the closure Module.
        # It is hard to implement device allocation for manifolds in other case.
        self.ball = ball
        if self.bias is not None:
            self.bias = geoopt.ManifoldParameter(self.bias, manifold=self.ball)
        self.reset_parameters()

    def forward(self, input):
        self.ball.assert_attached(input)
        return mobius_linear(input, weight=self.weight, bias=self.bias, ball=self.ball)

    @torch.no_grad()
    def reset_parameters(self):
        if self.bias is not None:
            self.bias.zero_()


def mobius_linear(input, weight, bias=None, *, ball: geoopt.PoincareBall):
    output = ball.mobius_matvec(weight, input)
    if bias is not None:
        output = ball.mobius_add(output, bias)
    return output
