import numbers
import itertools


__all__ = ["reshape_shape", "repeat"]


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
