def pairwise_distances(x, y=None, dim=0, squared=False, *, manifold):
    if y is None:
        y = x
    if dim < 0:
        if abs(dim) <= manifold.ndim:
            raise RuntimeError(
                "Incorrect usage of pairwise distances, dim should not touch manifold dimension"
            )
        x = x.unsqueeze(dim)
        y = y.unsqueeze(dim - 1)
    else:
        if manifold.ndim > 0 and (
            (x.dim() - dim) <= manifold.ndim or (y.dim() - dim) <= manifold.ndim
        ):
            raise RuntimeError(
                "Incorrect usage of pairwise distances, dim should not touch manifold dimension"
            )
        x = x.unsqueeze(dim + 1)
        y = y.unsqueeze(dim)
    if squared:
        output = manifold.dist2(x, y)
    else:
        output = manifold.dist(x, y)
    return output
