from ..base import ManifoldModule
from .hyperplanes import Distance2PoincareHyperplanes
from .centroids import WeightedPoincareCentroids
import torch.nn


class MobiusLinear(ManifoldModule):
    r"""
    Mobius Linear layer with nonlinearity that is based on distance to hyperplanes.

    The main idea befind the funtional form is the following.

    Conventional Euclidean Linear layer may be represented as

    .. math::

            y = Ax + b

    However, in regular neural networks linear operation is usually followed by batch norm and nonlinearity.
    In naive generalization of linear layer to hyperbolic space it is non obvious how to introduce these operations.
    Instead let's rewrite the operation in the following way:

    .. math::

        y_i = H_{A_i,b_i}(x),

    where :math:`H_{A_i,b_i}` is a distance to hyperplane. However, this is still not enough to fully satisfy
    the presence of batch norm and nonlinearity. WHat is missing is basis for output. By default euclidean
    linear layer considers canonical basis. We can introduce is explicitly simplifying the transition to
    hyperbolic generalization.

    ..math::

        y = \sum_i f_i(H_{A_i,b_i}(x)) e_i,

    where :math:`f_i` is an arbitrary nonlinearity. acting solely in :math:`\mathrm{R}` domain. Basis :math:`e_i`
    is weighted in a linear combination. Once we generalize distance to a hyperplane and linear combination,
    we are ready to write down the formula for hyperbolic linear layer.

    ..math::

        y = \left(\sum_i f_i(H_{A_i,b_i}(x)) \right) \odot
            \operatorname{Mid}\left\{
                \left( f_i(H_{A_i,b_i}(x)), e_i \right)
            \right\},

    where :math:`y, e_i \in H_\kappa^O`, :math:`x \in H_\kappa^I`

    Parameters
    ----------
    in_features : int
        input dimension
    out_features :
        output dimension
    nonlinearity :
        nonlinearity for distances. May be arbitrary, just pass batch norm or dropout there too.
    ball : Manifold
    ball_out : Optional[Manifold]
    num_basis : Optional[int]
        number of basis vectors
    """

    def __init__(
        self,
        in_features,
        out_features,
        nonlinearity=torch.nn.Identity(),
        *,
        ball,
        ball_out=None,
        num_basis=None,
    ):
        super().__init__()
        if ball_out is None:
            ball_out = ball
        if num_basis is None:
            num_basis = out_features
        self.num_basis = num_basis
        self.in_features = in_features
        self.out_features = out_features
        self.hyperplanes = Distance2PoincareHyperplanes(
            in_features, num_basis, ball=ball, scaled=True, squared=False,
        )
        self.basis = WeightedPoincareCentroids(
            out_features, num_basis, ball=ball_out, lincomb=True
        )
        self.nonlinearity = nonlinearity
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        self.basis.reset_parameters_identity()

    def forward(self, input):
        distances = self.hyperplanes(input)
        activations = self.nonlinearity(distances)
        return self.basis(activations)
