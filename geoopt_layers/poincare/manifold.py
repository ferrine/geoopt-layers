import geoopt
import torch


def method(name, alternative=None):
    def impl(self, *args, **kwargs):
        if self.disabled and alternative is None:
            return getattr(geoopt.Euclidean, name)(self, *args, **kwargs)
        elif self.disabled:
            return alternative(self, *args, **kwargs)
        else:
            return getattr(self.__base__, name)(self, *args, **kwargs)

    return impl


class PoincareBall(geoopt.PoincareBall):
    __base__ = geoopt.PoincareBall

    @property
    def reversible(self):
        return self.disabled

    def __init__(self, c=1.0, disable=False):
        geoopt.PoincareBall.__init__(self, c)
        self.ndim = 1
        self.disabled = disable

    _check_point_on_manifold = method("_check_point_on_manifold")
    _check_vector_on_tangent = method("_check_vector_on_tangent")
    dist = method(
        "dist",
        lambda s, x, y, *, keepdim=False, dim=-1: (x - y).norm(
            dim=dim, keepdim=keepdim
        ),
    )
    dist2 = method(
        "dist2",
        lambda s, x, y, *, keepdim=False, dim=-1: (x - y).sum(dim=dim, keepdim=keepdim),
    )
    egrad2rgrad = method("egrad2rgrad", lambda s, x, u, *, dim=-1: u)
    retr = method("retr", lambda s, x, u, *, dim=-1: x + u)
    projx = method("projx", lambda s, x, *, dim=-1: x)
    proju = method("proju", lambda s, x, u, *, dim=-1: x + u)

    def __inner(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor = None,
        *,
        keepdim=False,
        dim=-1
    ) -> torch.Tensor:
        if v is None:
            inner = u.pow(2).sum(dim=dim, keepdim=keepdim)
        else:
            inner = (u * v).sum(dim=dim, keepdim=keepdim)
        return inner

    inner = method("inner", __inner)

    def __norm(
        self, x: torch.Tensor, u: torch.Tensor, *, keepdim=False, dim=-1
    ) -> torch.Tensor:
        norm = u.norm(dim=dim, keepdim=keepdim)
        return norm

    norm = method("norm", __norm)
    expmap = method("expmap", lambda s, x, u, *, dim=-1: x + u)
    logmap = method("logmap", lambda s, x, y, *, dim=-1: y - x)
    transp = method("transp", lambda s, x, y, v, *, dim=-1: v)
    transp_follow_retr = method("transp_follow_retr", lambda s, x, u, v, *, dim=-1: v)
    transp_follow_expmap = method(
        "transp_follow_expmap", lambda s, x, u, v, *, dim=-1: v
    )
    expmap_transp = method("expmap_transp", lambda s, x, u, v, *, dim=-1: (x + u, v))
    retr_transp = method("retr_transp", lambda s, x, u, v, *, dim=-1: (x + u, v))

    # need alternative
    mobius_add = method("mobius_add", lambda s, x, y, *, dim=-1, project=True: x + y)
    mobius_sub = method("mobius_sub", lambda s, x, y, *, dim=-1, project=True: x - y)
    mobius_coadd = method(
        "mobius_coadd", lambda s, x, y, *, dim=-1, project=True: x + y
    )
    mobius_cosub = method(
        "mobius_cosub", lambda s, x, y, *, dim=-1, project=True: x - y
    )
    mobius_scalar_mul = method(
        "mobius_scalar_mul", lambda s, r, x, *, dim=-1, project=True: r * x
    )
    mobius_pointwise_mul = method(
        "mobius_pointwise_mul", lambda s, w, x, *, dim=-1, project=True: w * x
    )
    mobius_matvec = method(
        "mobius_matvec",
        lambda s, m, x, *, dim=-1, project=True: torch.tensordot(
            x, m, dims=([dim], [1])
        ),
    )
    geodesic = method(
        "geodesic", lambda s, t, x, y, *, dim=-1, project=True: torch.lerp(x, y, t)
    )

    def __geodesic_unit(self, t, x, u, *, dim=-1, project=True):
        u = u / u.norm(dim=dim, keepdim=True).clamp_min(1e-5)
        return x + u * t

    geodesic_unit = method("geodesic_unit", __geodesic_unit)
    lambda_x = method(
        "lambda_x",
        lambda s, x, *, keepdim=False, dim=-1: torch.full_like(x.narrow(dim, 0, 1), 2),
    )
    dist0 = method(
        "dist0", lambda s, x, *, dim=-1, keepdim=True: x.norm(dim=dim, keepdim=keepdim)
    )
    expmap0 = method("expmap0", lambda s, u, *, dim=-1, project=True: u)
    logmap0 = method("logmap0", lambda s, x, *, dim=-1: x)
    transp0 = method("transp0", lambda s, y, u, *, dim=-1: u)
    transp0back = method("transp0back", lambda s, y, u, *, dim=-1: u)
    gyration = method("gyration", lambda s, x, y, z, *, dim=-1: z)

    def __dist2plane(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        a: torch.Tensor,
        *,
        dim=-1,
        keepdim=False,
        signed=False
    ) -> torch.Tensor:
        a = a / a.norm(dim=dim, keepdim=True).clamp_min(1e-5)
        dist = ((x - p) * a).sum(dim=dim, keepdim=keepdim)
        if signed:
            return dist
        else:
            return dist.abs()

    dist2plane = method("dist2plane", __dist2plane)
    mobius_fn_apply = method("mobius_fn_apply", NotImplemented)
    mobius_fn_apply_chain = method("mobius_fn_apply_chain", NotImplemented)
    random_normal = method(
        "random_normal",
        lambda s, *size, mean=0, std=1, dtype=None, device=None: torch.randn(
            *size, dtype=dtype, device=device
        )
        * std
        + mean,
    )
    random = random_normal
    origin = method(
        "origin",
        lambda s, *size, dtype=None, device=None: torch.zeros(
            *size, dtype=dtype, device=device
        ),
    )

    def enable(self):
        self.disabled = False

    def disable(self):
        self.disabled = True

    def set_enabled(self, mode=True):
        self.disabled = not mode

    def set_disabled(self, mode=True):
        self.set_enabled(not mode)

    def extra_repr(self):
        return super().extra_repr() + ", disabled={}".format(self.disabled)


class PoincareBallExact(PoincareBall):
    __base__ = geoopt.PoincareBallExact

    reversible = True
    retr_transp = PoincareBall.expmap_transp
    transp_follow_retr = PoincareBall.transp_follow_expmap
    retr = PoincareBall.expmap

    def extra_repr(self):
        return "exact"
