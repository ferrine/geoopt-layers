import functools
import operator

import numbers
import itertools


__all__ = ["prod", "idx2sign", "reshape_shape", "repeat"]


def repeat(src, length):
    if src is None:
        return None
    if isinstance(src, numbers.Number):
        return list(itertools.repeat(src, length))
    if len(src) > length:
        return src[:length]
    if len(src) < length:
        return src + list(itertools.repeat(src[-1], length - len(src)))
    return src


def prod(items):
    return functools.reduce(operator.mul, items, 1)


def reshape_shape(shape, pattern):
    out_shape = []
    i = 0
    for s in pattern:
        if isinstance(s, int):
            break
        out_shape.append(shape[i])
        i += 1
    for s in pattern[i:]:
        if s == "*":
            break
        out_shape.append(pattern[i])
        i += 1

    for j, s in enumerate(reversed(pattern[i:])):
        out_shape.insert(i, shape[-j - 1])

    return out_shape


def idx2sign(idx, dim, neg=True):
    """
    Unify idx to be negative or positive, that helps in cases of broadcasting.

    Parameters
    ----------
    idx : int
        current index
    dim : int
        maximum dimension
    neg : bool
        indicate we need negative index

    Returns
    -------
    int
    """
    if neg:
        if idx < 0:
            return idx
        else:
            return (idx + 1) % -(dim + 1)
    else:
        return idx % dim
