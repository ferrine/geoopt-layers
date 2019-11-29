def pairwise_distances(x, y=None, dim=0, squared=False, *, manifold):
    if y is None:
        y = x
    if dim == x.dim() or dim == y.dim():
        raise RuntimeError("Dim should be not the last one")
    if dim < 0:
        x = x.unsqueeze(dim)
        y = y.unsqueeze(dim - 1)
    else:
        x = x.unsqueeze(dim + 1)
        y = y.unsqueeze(dim)
    if squared:
        output = manifold.dist2(x, y)
    else:
        output = manifold.dist(x, y)
    return output
