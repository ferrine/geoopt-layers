import geoopt
import torch
from typing import Tuple, Optional


class PoincareBall(geoopt.PoincareBall):
    __base__ = geoopt.PoincareBall

    @property
    def reversible(self):
        return self.disabled

    def __init__(self, c=1.0, disable=False):
        geoopt.PoincareBall.__init__(self, c)
        self.ndim = 1
        self.disabled = disable

    def _check_point_on_manifold(
        self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5, dim=-1
    ) -> Tuple[bool, Optional[str]]:
        if self.disabled:
            return True, None
        else:
            return super()._check_point_on_manifold(x=x, atol=atol, rtol=rtol, dim=-1)

    def _check_vector_on_tangent(
        self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5, dim=-1
    ) -> Tuple[bool, Optional[str]]:
        if self.disabled:
            return True, None
        else:
            return super()._check_vector_on_tangent(
                x=x, u=u, atol=atol, rtol=rtol, dim=dim
            )

    def dist(
        self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False, dim=-1
    ) -> torch.Tensor:
        if self.disabled:
            return (x - y).norm(dim=dim, keepdim=keepdim)
        else:
            return super().dist(x=x, y=y, dim=dim, keepdim=keepdim)

    def dist2(
        self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False, dim=-1
    ) -> torch.Tensor:
        if self.disabled:
            return (x - y).pow(2).sum(dim=dim, keepdim=keepdim)
        else:
            return super().dist2(x=x, y=y, dim=dim, keepdim=keepdim)

    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        if self.disabled:
            return u
        else:
            return super().egrad2rgrad(x=x, u=u, dim=dim)

    def retr(self, x: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        if self.disabled:
            return x + u
        else:
            return super().retr(x=x, u=u, dim=dim)

    def projx(self, x: torch.Tensor, *, dim=-1) -> torch.Tensor:
        if self.disabled:
            return x
        else:
            return super().projx(x=x, dim=dim)

    def proju(self, x: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        if self.disabled:
            return u
        else:
            return super().proju(x=x, u=u, dim=dim)

    def inner(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor = None,
        *,
        keepdim=False,
        dim=-1
    ) -> torch.Tensor:
        if self.disabled:
            if v is None:
                inner = u.pow(2).sum(dim=dim, keepdim=keepdim)
            else:
                inner = (u * v).sum(dim=dim, keepdim=keepdim)
            return inner
        else:
            return super().inner(x=x, u=u, v=v, dim=dim, keepdim=keepdim)

    def norm(
        self, x: torch.Tensor, u: torch.Tensor, *, keepdim=False, dim=-1
    ) -> torch.Tensor:
        if self.disabled:
            return u.norm(dim=dim, keepdim=keepdim)
        else:
            return super().norm(x=x, u=u, keepdim=keepdim, dim=dim)

    def expmap(
        self, x: torch.Tensor, u: torch.Tensor, *, project=True, dim=-1
    ) -> torch.Tensor:
        if self.disabled:
            return x + u
        else:
            return super().expmap(x=x, u=u, project=project, dim=dim)

    def logmap(self, x: torch.Tensor, y: torch.Tensor, *, dim=-1) -> torch.Tensor:
        if self.disabled:
            return y - x
        else:
            return super().logmap(x=x, y=y, dim=dim)

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, *, dim=-1):
        if self.disabled:
            return v
        else:
            return super().transp(x=x, y=y, v=v, dim=dim)

    def transp_follow_retr(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, *, dim=-1
    ) -> torch.Tensor:
        if self.disabled:
            return v
        else:
            return super().transp_follow_retr(x=x, u=u, v=v, dim=dim)

    def transp_follow_expmap(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, *, dim=-1, project=True
    ) -> torch.Tensor:
        if self.disabled:
            return v
        else:
            return super().transp_follow_expmap(x=x, u=u, v=v, dim=dim, project=project)

    def expmap_transp(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, *, dim=-1, project=True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.disabled:
            return x + u, v
        else:
            return super().expmap_transp(x=x, u=u, v=v, dim=dim, project=project)

    def retr_transp(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, *, dim=-1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.disabled:
            return x + u, v
        else:
            return super().retr_transp(x=x, u=u, v=v, dim=dim)

    def mobius_add(
        self, x: torch.Tensor, y: torch.Tensor, *, dim=-1, project=True
    ) -> torch.Tensor:
        if self.disabled:
            return x + y
        else:
            return super().mobius_add(x=x, y=y, dim=dim, project=project)

    def mobius_sub(
        self, x: torch.Tensor, y: torch.Tensor, *, dim=-1, project=True
    ) -> torch.Tensor:
        if self.disabled:
            return x - y
        else:
            return super().mobius_sub(x=x, y=y, dim=dim, project=project)

    def mobius_coadd(
        self, x: torch.Tensor, y: torch.Tensor, *, dim=-1, project=True
    ) -> torch.Tensor:
        if self.disabled:
            return x + y
        else:
            return super().mobius_coadd(x=x, y=y, dim=dim, project=project)

    def mobius_cosub(
        self, x: torch.Tensor, y: torch.Tensor, *, dim=-1, project=True
    ) -> torch.Tensor:
        if self.disabled:
            return x - y
        else:
            return super().mobius_cosub(x=x, y=y, dim=dim, project=project)

    def mobius_scalar_mul(
        self, r: torch.Tensor, x: torch.Tensor, *, dim=-1, project=True
    ) -> torch.Tensor:
        if self.disabled:
            return r * x
        else:
            return super().mobius_scalar_mul(r=r, x=x, dim=dim, project=project)

    def mobius_pointwise_mul(
        self, w: torch.Tensor, x: torch.Tensor, *, dim=-1, project=True
    ) -> torch.Tensor:
        if self.disabled:
            return w * x
        else:
            return self.mobius_pointwise_mul(w=w, x=x, dim=dim, project=project)

    def mobius_matvec(
        self, m: torch.Tensor, x: torch.Tensor, *, dim=-1, project=True
    ) -> torch.Tensor:
        if self.disabled:
            return torch.tensordot(x, m, dims=([dim], [1]))
        else:
            return super().mobius_matvec(m=m, x=x, dim=dim, project=project)

    def geodesic(
        self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, *, dim=-1
    ) -> torch.Tensor:
        if self.disabled:
            return torch.lerp(x, y, t)
        else:
            return super().geodesic(t=t, x=x, y=y, dim=dim)

    def geodesic_unit(self, t, x, u, *, dim=-1, project=True):
        if self.disabled:
            u = u / u.norm(dim=dim, keepdim=True).clamp_min(1e-5)
            return x + u * t
        else:
            return super().geodesic_unit(t=t, x=x, u=u, dim=dim, project=project)

    def lambda_x(self, x: torch.Tensor, *, dim=-1, keepdim=False) -> torch.Tensor:
        if self.disabled:
            lam = torch.full_like(x.narrow(dim, 0, 1), 2)
            if not keepdim:
                lam = lam.squeeze(dim)
            return lam
        else:
            return super().lambda_x(x=x, dim=dim, keepdim=keepdim)

    def dist0(self, x: torch.Tensor, *, dim=-1, keepdim=False) -> torch.Tensor:
        if self.disabled:
            return x.norm(dim=dim, keepdim=keepdim)
        else:
            return super().dist0(x=x, dim=dim, keepdim=keepdim)

    def expmap0(self, u: torch.Tensor, *, dim=-1, project=True) -> torch.Tensor:
        if self.disabled:
            return u
        else:
            return super().expmap0(u=u, dim=dim, project=project)

    def logmap0(self, x: torch.Tensor, *, dim=-1) -> torch.Tensor:
        if self.disabled:
            return x
        else:
            return super().logmap0(x=x, dim=dim)

    def transp0(self, y: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        if self.disabled:
            return u
        else:
            return super().transp0(y=y, u=u, dim=dim)

    def transp0back(self, y: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        if self.disabled:
            return u
        else:
            return super().transp0back(y=y, u=u, dim=dim)

    def gyration(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, *, dim=-1
    ) -> torch.Tensor:
        if self.disabled:
            return z
        else:
            return super().gyration(x=x, y=y, z=z, dim=dim)

    def dist2plane(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        a: torch.Tensor,
        *,
        dim=-1,
        keepdim=False,
        signed=False
    ) -> torch.Tensor:
        if self.disabled:
            a = a / a.norm(dim=dim, keepdim=True).clamp_min(1e-5)
            dist = ((x - p) * a).sum(dim=dim, keepdim=keepdim)
            if signed:
                return dist
            else:
                return dist.abs()
        else:
            return super().dist2plane(
                x=x, p=p, a=a, dim=dim, keepdim=keepdim, signed=signed
            )

    def mobius_fn_apply(
        self, fn: callable, x: torch.Tensor, *args, dim=-1, project=True, **kwargs
    ) -> torch.Tensor:
        if self.disabled:
            return fn(x, *args, **kwargs)
        else:
            return super().mobius_fn_apply(
                fn=fn, x=x, *args, dim=dim, project=project, **kwargs
            )

    def mobius_fn_apply_chain(
        self, x: torch.Tensor, *fns: callable, project=True, dim=-1
    ) -> torch.Tensor:
        if self.disabled:
            for fn in fns:
                x = fn(x)
            return x
        else:
            return super().mobius_fn_apply(fns=fns, x=x, dim=dim, project=project)

    def enable(self):
        self.disabled = False
        return self

    def disable(self):
        self.disabled = True
        return self

    def set_enabled(self, mode=True):
        self.disabled = not mode
        return self

    def set_disabled(self, mode=True):
        self.set_enabled(not mode)
        return self

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
